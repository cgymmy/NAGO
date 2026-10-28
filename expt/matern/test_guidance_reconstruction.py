"""Generate DPS-guided samples for a controlled Matérn target."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from expt.matern.main import get_flow_model, normalize_checkpoint_state_dict
from models.fno import get_model as get_fno
from src.functional import GPPrior
from src.utilities import seed_all


def sample_guided(
    model,
    truth,
    n_samples,
    batch_size,
    n_eval,
    loss_scale,
    scale_mode,
    subx,
    device,
):
    generated = []
    loss_trajectories = []
    started = time.perf_counter()
    for batch_start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - batch_start)
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
        step_scale = (
            loss_scale / n_eval if scale_mode == "total" else loss_scale
        )
        dims = [1, truth.shape[-2], truth.shape[-1]]
        xt = model.gp.sample(n_samples=size).reshape(size, *dims)
        losses = []
        for step in range(n_eval):
            xt = xt.detach().requires_grad_(True)
            hat_xt = model.tweedie(xt, ts[step])
            per_sample_loss = torch.mean(
                (
                    truth[None, ::subx, ::subx]
                    - hat_xt[:, 0, ::subx, ::subx]
                ).square(),
                dim=(1, 2),
            )
            grad = torch.autograd.grad(step_scale * per_sample_loss.sum(), xt)[0]
            xt = xt - grad
            losses.append(per_sample_loss.detach().cpu())
            xt = model.sample_one_step(
                xt, ts[step], ts[step + 1], device, method="rk4"
            )
        generated.append(xt[:, 0].detach().cpu())
        loss_trajectories.append(torch.stack(losses, dim=1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        torch.cat(generated, dim=0),
        torch.cat(loss_trajectories, dim=0),
        time.perf_counter() - started,
    )


def sample_target(config, n_samples, seed, device):
    seed_all(seed)
    prior = GPPrior(
        dim=int(config.datasets.dim),
        L=float(config.datasets.L),
        res=int(config.sample_dims[-1]),
        nu=float(config.datasets.nu),
        ell=float(config.datasets.ell),
        device=device,
    )
    return prior.sample(n_samples, normalize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--loss-scale", type=float, default=100.0)
    parser.add_argument(
        "--scale-mode",
        choices=["per_step", "total"],
        default="per_step",
        help=(
            "With 'total', multiply loss-scale by each integration dt so the "
            "trajectory-integrated guidance strength is independent of n-eval."
        ),
    )
    parser.add_argument("--subx", type=int, default=2)
    parser.add_argument("--truth-seed", type=int, default=20260728)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    resolution = int(config.sample_dims[-1])
    modes = [int(value) for value in config.fno.n_modes]
    if resolution != 64 or modes != [64, 64]:
        raise ValueError(
            f"Expected resolution=64 and modes=[64, 64], got "
            f"resolution={resolution}, modes={modes}"
        )

    model = get_flow_model(config.noise, get_fno(config.fno), device)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint["model"])
    )
    model.eval()
    model.requires_grad_(False)

    truth = sample_target(config, 1, args.truth_seed, device)[0]
    seed_all(args.sample_seed)
    generated, loss_trajectories, elapsed = sample_guided(
        model=model,
        truth=truth,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_eval=args.n_eval,
        loss_scale=args.loss_scale,
        scale_mode=args.scale_mode,
        subx=args.subx,
        device=device,
    )

    truth_cpu = truth.cpu()
    observed = torch.zeros_like(truth_cpu, dtype=torch.bool)
    observed[::args.subx, ::args.subx] = True
    observed_errors = (
        generated[:, observed] - truth_cpu[observed]
    ).square().mean(dim=1)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "noise_type": str(config.noise.noise_type),
        "noise_nu": (
            float(config.noise.nu)
            if str(config.noise.noise_type) == "matern"
            else None
        ),
        "noise_ell": (
            float(config.noise.ell)
            if str(config.noise.noise_type) == "matern"
            else None
        ),
        "target_nu": float(config.datasets.nu),
        "target_ell": float(config.datasets.ell),
        "resolution": resolution,
        "fno_modes": modes,
        "n_samples": args.n_samples,
        "n_eval": args.n_eval,
        "loss_scale": args.loss_scale,
        "scale_mode": args.scale_mode,
        "observation_fraction": float(observed.float().mean()),
        "truth_seed": args.truth_seed,
        "sample_seed": args.sample_seed,
        "observed_mse_per_generated_sample": float(observed_errors.mean()),
        "observed_mse_per_generated_sample_std": float(
            observed_errors.std(unbiased=False)
        ),
        "final_unscaled_observation_loss": float(
            loss_trajectories[:, -1].mean()
        ),
        "wall_seconds": elapsed,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(result, output_file, indent=2)
    os.makedirs(os.path.dirname(args.samples_output) or ".", exist_ok=True)
    torch.save(
        {
            "samples": generated,
            "truth": truth_cpu,
            "loss_trajectories": loss_trajectories,
            "result": result,
        },
        args.samples_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
