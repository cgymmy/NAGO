"""Evaluate spectral-density bias of an SNS checkpoint.

Example (on the training cluster):
  python expt/sns/test_spectral_bias.py --config configs/sns_configs.yaml \
    --checkpoint logs/sns/sns1/latest.pt --n-samples 2000 --n-eval 100
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from expt.sns.main import get_flow_model, get_many_forecasting_dataloader, normalize_checkpoint_state_dict
from expt.sns.metrics import get_energy_spectrum
from src.utilities import load_config, seed_all


def log_mse(pred, target, eps=1e-12):
    pred = np.clip(np.asarray(pred), eps, None)
    target = np.clip(np.asarray(target), eps, None)
    return float(np.mean((np.log(pred) - np.log(target)) ** 2))


def vorticity_moments(x, device, distributed):
    """Return global mean, variance, skewness and kurtosis of vorticity."""
    x = x.to(device=device, dtype=torch.float64)
    count = torch.tensor(float(x.numel()), dtype=torch.float64, device=device)
    raw = torch.stack([(x ** p).sum() for p in range(1, 5)])
    if distributed:
        torch.distributed.all_reduce(count)
        torch.distributed.all_reduce(raw)
    m1, m2, m3, m4 = raw / count
    variance = torch.clamp(m2 - m1 ** 2, min=1e-30)
    central3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    central4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return {
        'mean': m1.item(),
        'variance': variance.item(),
        'skewness': (central3 / variance.pow(1.5)).item(),
        'kurtosis': (central4 / variance.pow(2)).item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/sns_configs.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='data/sns')
    p.add_argument('--processed-data-root', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n-samples', type=int, default=2000)
    p.add_argument('--gen-batch-size', type=int, default=32)
    p.add_argument('--n-eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42,
                   help='base random seed; distributed ranks use seed + rank')
    p.add_argument('--sample-resolution', type=int, default=None,
                   help='override generation resolution, e.g. 32')
    p.add_argument('--max-truth', type=int, default=None)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group('nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    seed_all(args.seed + rank)
    config = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Use the exact configuration stored with the checkpoint. This avoids
    # silently evaluating weights with a changed FNO mode/resolution config.
    eval_config = ckpt.get('config', config)
    sample_resolution = args.sample_resolution or int(eval_config.sample_dims[-1])
    if args.sample_resolution is not None:
        eval_config.noise.res = sample_resolution
        eval_config.sample_dims = [int(eval_config.sample_dims[0]), sample_resolution, sample_resolution]
    model_cfg = eval_config.fno if eval_config.model_type == 'fno' else getattr(eval_config, eval_config.model_type)
    model = get_flow_model(eval_config.noise, model_cfg, device)
    model.load_state_dict(normalize_checkpoint_state_dict(ckpt['model']))
    model.eval()

    data_cfg = config.datasets
    root = args.processed_data_root or getattr(data_cfg, 'processed_root', None)
    files = [os.path.join(args.data_root, f'data_file0{i}.pt')
             for i in range(1, data_cfg.num_dataset + 1)]
    requested_n = min(args.n_samples, args.max_truth or args.n_samples)
    if sample_resolution > int(data_cfg.hi_size):
        # For super-resolution evaluation, use the native 256x256 raw data as
        # truth instead of upsampling the cached 128x128 tensor. mmap keeps the
        # full source tensor off resident memory; each rank materializes only
        # its own test subset.
        data_raw, _ = torch.load(
            files[0], map_location='cpu', mmap=True, weights_only=False
        )
        if data_cfg.time_lag > 0:
            data_raw = data_raw[:, data_cfg.time_lag:, ...]
        flat = data_raw.reshape(-1, data_raw.shape[-2], data_raw.shape[-1])
        if data_cfg.subsampling_ratio:
            flat = flat[:int(data_cfg.subsampling_ratio * flat.shape[0])]
        num_train = int(flat.shape[0] * data_cfg.train_test_split)
        available = flat.shape[0] - num_train
        n = min(requested_n, available)
        start = (n * rank) // world_size
        end = (n * (rank + 1)) // world_size
        truth = (flat[num_train + start:num_train + end] / 3.0679163932800293).contiguous()
        if truth.shape[-1] != sample_resolution:
            truth = torch.nn.functional.interpolate(
                truth[:, None], size=(sample_resolution, sample_resolution), mode='bilinear'
            )[:, 0]
        truth = truth.to(device)
        del data_raw, flat
    else:
        _, test_loaders, _, _, _ = get_many_forecasting_dataloader(
            files, data_cfg.time_lag, data_cfg.lo_size, data_cfg.hi_size,
            batch_size=min(int(data_cfg.batch_size), args.n_samples),
            train_test_split=data_cfg.train_test_split,
            subsampling_ratio=data_cfg.subsampling_ratio,
            num_workers=0, processed_data_root=root)
        truth = torch.cat([batch[0] for loader in test_loaders for batch in loader], dim=0)
        truth = truth[:requested_n, 0]
        if truth.shape[-1] != sample_resolution:
            truth = torch.nn.functional.interpolate(
                truth[:, None], size=(sample_resolution, sample_resolution), mode='bilinear'
            )[:, 0]
        n = min(requested_n, truth.shape[0])
        # Deterministic contiguous partition; every rank only holds its own batch.
        start = (n * rank) // world_size
        end = (n * (rank + 1)) // world_size
        truth = truth[start:end].to(device)
    local_n = end - start

    generated_parts = []
    batch_offsets = range(0, local_n, args.gen_batch_size)
    num_batches = (local_n + args.gen_batch_size - 1) // args.gen_batch_size
    with torch.no_grad():
        for batch_index, offset in enumerate(batch_offsets, start=1):
            size = min(args.gen_batch_size, local_n - offset)
            generated_parts.append(model.sample(
                size, args.n_eval, eval_config.sample_dims, device,
                progress=rank == 0,
                progress_desc=f'ODE batch {batch_index}/{num_batches} (rank 0/{world_size})',
            )[:, 0])
    generated = torch.cat(generated_parts, dim=0) if generated_parts else truth[:0]
    kvals, truth_w, truth_v = get_energy_spectrum(truth.cpu())
    _, generated_w, generated_v = get_energy_spectrum(generated.cpu())
    truth_moments = vorticity_moments(truth, device, distributed)
    generated_moments = vorticity_moments(generated, device, distributed)
    if distributed:
        def aggregate(values):
            value = torch.as_tensor(np.nan_to_num(values), dtype=torch.float64, device=device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            return (value / world_size).cpu().numpy()
        truth_w, truth_v = aggregate(truth_w), aggregate(truth_v)
        generated_w, generated_v = aggregate(generated_w), aggregate(generated_v)
        torch.distributed.barrier()
    if rank != 0:
        if distributed:
            torch.distributed.destroy_process_group()
        return
    result = {
        'checkpoint': args.checkpoint,
        'n_samples': n,
        'world_size': world_size,
        'gen_batch_size': args.gen_batch_size,
        'n_eval': args.n_eval,
        'resolution': int(truth.shape[-1]),
        'log_mse_enstrophy_spectrum': log_mse(generated_w, truth_w),
        'log_mse_energy_spectrum': log_mse(generated_v, truth_v),
        'mean_relative_enstrophy_bias': float(np.mean(np.abs(generated_w - truth_w) /
                                                       np.maximum(np.abs(truth_w), 1e-12))),
        'mean_relative_energy_bias': float(np.mean(np.abs(generated_v - truth_v) /
                                                   np.maximum(np.abs(truth_v), 1e-12))),
        'truth_vorticity_moments': truth_moments,
        'generated_vorticity_moments': generated_moments,
        'vorticity_moment_absolute_errors': {
            key: abs(generated_moments[key] - truth_moments[key])
            for key in truth_moments
        },
        'k_values': kvals.tolist(),
        'truth_enstrophy_spectrum': truth_w.tolist(),
        'generated_enstrophy_spectrum': generated_w.tolist(),
        'truth_energy_spectrum': truth_v.tolist(),
        'generated_energy_spectrum': generated_v.tolist(),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
