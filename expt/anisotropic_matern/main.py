"""Train Gaussian-source flow matching on anisotropic Matérn fields."""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.fno import get_model as get_fno
from src.functional import FFM, GNPrior, GPPrior, interpolation
from src.random_field import AnisotropicMaternFieldGenerator
from src.utilities import (
    count_parameters,
    get_optimizer,
    get_scheduler,
    load_config,
    seed_all,
)


class TensorFieldDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def setup_distributed(requested_device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return distributed, rank, local_rank, world_size, device


def configure_cuda():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def reduce_mean(value, device, world_size):
    reduced = torch.as_tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
    return reduced.item()


def reduce_max(value, device, world_size):
    reduced = torch.as_tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced.item()


def flow_state_dict(model):
    return {
        key.replace("model.module.", "model.", 1): value
        for key, value in model.state_dict().items()
    }


def normalize_checkpoint_state_dict(state_dict):
    return {
        key.replace("model.module.", "model.", 1): value
        for key, value in state_dict.items()
    }


def build_flow(config, device):
    if str(config.noise.intp_type) != "linear":
        raise ValueError("Only linear interpolation is supported in this experiment.")
    if str(config.model_type) != "fno":
        raise ValueError("The anisotropic baseline currently expects an FNO.")

    network = get_fno(config.fno)
    noise_type = str(config.noise.noise_type)
    if noise_type == "gaussian":
        noise = GNPrior(
            dim=int(config.noise.dim),
            res=int(config.noise.res),
            device=device,
        )
    elif noise_type == "matern":
        noise = GPPrior(
            dim=int(config.noise.dim),
            L=float(config.noise.L),
            res=int(config.noise.res),
            nu=float(config.noise.nu),
            ell=float(config.noise.ell),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported source noise: {noise_type!r}")
    return FFM(network, noise, interpolation()).to(device)


def experiment_name(config):
    target = config.target
    modes = "x".join(str(int(value)) for value in config.fno.n_modes)
    noise_type = str(config.noise.noise_type)
    noise_name = noise_type
    if noise_type == "matern":
        noise_name += (
            f"_nu{float(config.noise.nu):g}"
            f"_ell{float(config.noise.ell):g}"
        )
    return (
        f"{int(target.res)}_fno_mode{modes}_{noise_name}"
        f"_aniso_matern_nu{float(target.nu):g}"
        f"_ellx{float(target.ell_x):g}_elly{float(target.ell_y):g}"
        f"_theta{float(target.theta_deg):g}"
    )


def generate_target_shard(target_cfg, count, seed, device, chunk_size=256):
    """Generate one independent rank-local dataset shard."""
    generator = AnisotropicMaternFieldGenerator(
        L=float(target_cfg.L),
        res=int(target_cfg.res) - 1,
        nu=float(target_cfg.nu),
        ell_x=float(target_cfg.ell_x),
        ell_y=float(target_cfg.ell_y),
        theta=math.radians(float(target_cfg.theta_deg)),
        device=device,
    )
    parts = []
    for start in range(0, count, chunk_size):
        size = min(chunk_size, count - start)
        sample_seed = int(seed + start)
        fields = generator.generate(
            num_realizations=size,
            seed=sample_seed,
            normalize=False,
        )[3]
        parts.append(fields.unsqueeze(1).cpu())
    data = torch.cat(parts, dim=0)
    data = (data - data.mean()) / data.std().clamp_min(1e-12)
    return data.contiguous()


def make_loader(data, batch_size, shuffle, num_workers, drop_last):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(TensorFieldDataset(data), **kwargs)


def amp_settings(value, device):
    value = str(value).lower()
    if device.type != "cuda" or value in {"none", "false", "fp32"}:
        return False, None
    if value == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but is not supported by this GPU.")
        return True, torch.bfloat16
    if value == "fp16":
        return True, torch.float16
    raise ValueError(f"Unsupported train.amp value: {value}")


def autocast_context(enabled, dtype, device):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def flow_loss(model, target):
    t = torch.rand(target.size(0), dtype=torch.float32, device=target.device)
    interpolated, velocity = model.simulate(t, target)
    prediction = model(t, interpolated)
    return (prediction.float() - velocity.float()).square().mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="expt/anisotropic_matern/config.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logdir", default="logs/anisotropic_matern")
    parser.add_argument("--savename", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-valid", type=int, default=None)
    parser.add_argument(
        "--amp", choices=["fp32", "fp16", "bf16"], default=None
    )
    parser.add_argument(
        "--noise-type", choices=["gaussian", "matern"], default=None
    )
    parser.add_argument("--noise-nu", type=float, default=None)
    parser.add_argument("--noise-ell", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--fno-modes", type=int, nargs=2, default=None)
    parser.add_argument("--ell-x", type=float, default=None)
    parser.add_argument("--ell-y", type=float, default=None)
    parser.add_argument("--theta-deg", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset-chunk-size",
        type=int,
        default=256,
        help="Maximum number of fields generated in one FFT batch.",
    )
    return parser.parse_args()


def apply_overrides(config, args):
    if args.max_iter is not None:
        config.train.max_iter = args.max_iter
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_train is not None:
        config.train.num_train = args.num_train
    if args.num_valid is not None:
        config.train.num_valid = args.num_valid
    if args.amp is not None:
        config.train.amp = args.amp
    if args.noise_type is not None:
        config.noise.noise_type = args.noise_type
    if args.noise_nu is not None:
        config.noise.nu = args.noise_nu
    if args.noise_ell is not None:
        config.noise.ell = args.noise_ell
    if args.resolution is not None:
        config.target.res = args.resolution
        config.noise.res = args.resolution
        config.sample_dims[-2:] = [args.resolution, args.resolution]
    if args.fno_modes is not None:
        config.fno.n_modes = list(args.fno_modes)
    if args.ell_x is not None:
        config.target.ell_x = args.ell_x
    if args.ell_y is not None:
        config.target.ell_y = args.ell_y
    if args.theta_deg is not None:
        config.target.theta_deg = args.theta_deg


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed(
        args.device
    )
    is_main = rank == 0
    configure_cuda()

    config = load_config(args.config)
    apply_overrides(config, args)
    if int(config.train.batch_size) % world_size:
        raise ValueError("Global batch size must be divisible by world size.")
    if int(config.train.num_train) % world_size:
        raise ValueError("train.num_train must be divisible by world size.")
    if int(config.train.num_valid) % world_size:
        raise ValueError("train.num_valid must be divisible by world size.")

    per_device_batch = int(config.train.batch_size) // world_size
    if per_device_batch < 1:
        raise ValueError("Per-device batch size must be at least one.")
    savename = args.savename or experiment_name(config)
    output_dir = os.path.join(args.logdir, savename)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    # Independent rank-local shards avoid eight full copies of the synthetic
    # dataset while preserving the requested global sample count per epoch.
    data_start = time.perf_counter()
    train_count = int(config.train.num_train) // world_size
    valid_count = int(config.train.num_valid) // world_size
    train_data = generate_target_shard(
        config.target,
        train_count,
        int(config.train.seed) + rank * 1_000_000,
        device,
        args.dataset_chunk_size,
    )
    valid_data = generate_target_shard(
        config.target,
        valid_count,
        int(config.train.seed) + 100_000_000 + rank * 1_000_000,
        device,
        args.dataset_chunk_size,
    )
    train_loader = make_loader(
        train_data,
        per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = make_loader(
        valid_data,
        int(config.train.valid_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Dataset randomness differs across ranks; subsequent flow noise also must.
    seed_all(int(config.train.seed) + 10_000 + rank)
    model = build_flow(config, device)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(
            normalize_checkpoint_state_dict(resume_checkpoint["model"])
        )
    if distributed:
        model.model = DDP(
            model.model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    global_step = 0
    if resume_checkpoint is not None:
        global_step = int(resume_checkpoint.get("step", 0))
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])

    amp_enabled, amp_dtype = amp_settings(config.train.amp, device)
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    writer = SummaryWriter(output_dir) if is_main else None

    if is_main:
        print(
            f"output={output_dir}\n"
            f"GPUs={world_size}, global_batch={config.train.batch_size}, "
            f"per_gpu_batch={per_device_batch}, amp={config.train.amp}\n"
            f"rank-local train/valid={train_count}/{valid_count}, "
            f"data_generation_seconds={time.perf_counter() - data_start:.2f}\n"
            f"parameters={count_parameters(model)}",
            flush=True,
        )

    def validate():
        model.eval()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        sample_count = torch.zeros((), dtype=torch.float64, device=device)
        max_batches = min(
            int(config.train.valid_max_batch or len(valid_loader)),
            len(valid_loader),
        )
        with torch.no_grad():
            for index, target in enumerate(valid_loader):
                if index >= max_batches:
                    break
                target = target.to(device, non_blocking=True)
                # Validation is deliberately FP32. Complex-half FFT can
                # occasionally overflow for an unlucky (t, noise) draw.
                loss = flow_loss(model, target)
                loss_sum += loss.detach().double() * target.size(0)
                sample_count += target.size(0)
        if distributed:
            dist.all_reduce(loss_sum)
            dist.all_reduce(sample_count)
        model.train()
        return (loss_sum / sample_count).item()

    def save_checkpoint(validation_loss=None):
        if not is_main:
            return
        checkpoint = {
            "config": config,
            "model": flow_state_dict(model),
            "step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "avg_val_loss": validation_loss,
            "world_size": world_size,
            "amp": str(config.train.amp),
        }
        torch.save(checkpoint, os.path.join(output_dir, "latest.pt"))
        if global_step and global_step % int(config.train.save_freq) == 0:
            torch.save(
                checkpoint, os.path.join(output_dir, f"{global_step}.pt")
            )

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    last_log_step = global_step
    last_log_time = time.perf_counter()
    latest_validation = None
    try:
        while global_step < int(config.train.max_iter):
            model.train()
            for target in train_loader:
                if global_step >= int(config.train.max_iter):
                    break
                target = target.to(device, non_blocking=True)
                with autocast_context(amp_enabled, amp_dtype, device):
                    loss = flow_loss(model, target)
                local_nonfinite = (~torch.isfinite(loss)).to(
                    dtype=torch.int32, device=device
                )
                if distributed:
                    dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX)
                if local_nonfinite.item() and is_main:
                    print(
                        f"step={global_step} nonfinite_rank_loss_zeroed",
                        flush=True,
                    )
                # Every rank must execute backward after a DDP forward. A rank
                # with a rare complex-half overflow contributes zero gradient;
                # finite ranks still contribute their normal gradients.
                backward_loss = torch.nan_to_num(
                    loss, nan=0.0, posinf=0.0, neginf=0.0
                )
                if use_scaler:
                    scaler.scale(backward_loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    backward_loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(), float(config.train.max_grad_norm)
                )
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(config.train.log_freq) == 0:
                    train_loss = reduce_mean(
                        torch.nan_to_num(loss.detach()), device, world_size
                    )
                    mean_grad = reduce_mean(grad_norm.detach(), device, world_size)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-12)
                    step_delta = global_step - last_log_step
                    throughput = (
                        step_delta * int(config.train.batch_size) / elapsed
                    )
                    max_memory_gib = (
                        reduce_max(
                            torch.cuda.max_memory_allocated(device),
                            device,
                            world_size,
                        )
                        / 2**30
                        if device.type == "cuda"
                        else 0.0
                    )
                    if is_main:
                        writer.add_scalar("train/loss", train_loss, global_step)
                        writer.add_scalar("train/grad", mean_grad, global_step)
                        writer.add_scalar(
                            "train/lr",
                            optimizer.param_groups[0]["lr"],
                            global_step,
                        )
                        writer.add_scalar(
                            "train/samples_per_second", throughput, global_step
                        )
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={train_loss:.6f} grad={mean_grad:.3f} "
                            f"samples/s={throughput:.1f} "
                            f"max_memory_gib={max_memory_gib:.2f}",
                            flush=True,
                        )
                    last_log_time = now
                    last_log_step = global_step

                if global_step % int(config.train.val_freq) == 0:
                    latest_validation = validate()
                    if str(config.train.scheduler.type) == "plateau":
                        scheduler.step(latest_validation)
                    else:
                        scheduler.step()
                    if is_main:
                        writer.add_scalar(
                            "valid/loss", latest_validation, global_step
                        )
                        print(
                            f"step={global_step} valid_loss="
                            f"{latest_validation:.6f}",
                            flush=True,
                        )
                    save_checkpoint(latest_validation)
            epoch += 1

        save_checkpoint(latest_validation)
        if distributed:
            dist.barrier(device_ids=[local_rank])
        if is_main:
            print(f"Training finished at step {global_step}.", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
