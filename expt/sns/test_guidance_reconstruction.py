"""Evaluate DPS-guided SNS reconstruction for a single trained checkpoint."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from expt.sns.main import get_flow_model, normalize_checkpoint_state_dict
from expt.sns.metrics import get_energy_spectrum
from src.utilities import seed_all


def load_samples(path, split):
    item = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(item, dict):
        samples = item[split]
    elif isinstance(item, (tuple, list)):
        samples = item[0]
    else:
        samples = item
    if samples.ndim == 4 and samples.shape[1] == 1:
        samples = samples[:, 0]
    if samples.ndim != 3:
        raise ValueError(
            f"Expected samples shaped [N,H,W], got {tuple(samples.shape)}"
        )
    return samples.float().contiguous()


def load_truth(path, split, index):
    samples = load_samples(path, split)
    truth = samples[index]
    if truth.ndim != 2:
        raise ValueError(f"Expected a 2D truth field, got shape {tuple(truth.shape)}")
    return truth.float().contiguous()


def load_reference(paths, split, max_samples):
    per_file = int(np.ceil(max_samples / len(paths)))
    parts = [load_samples(path, split)[:per_file] for path in paths]
    return torch.cat(parts, dim=0)[:max_samples].contiguous()


def log_mse(predicted, target, eps=1e-12):
    predicted = np.clip(np.asarray(predicted), eps, None)
    target = np.clip(np.asarray(target), eps, None)
    return float(np.mean((np.log(predicted) - np.log(target)) ** 2))


def sample_guided(model, truth, n_samples, batch_size, n_eval, loss_scale, subx, device):
    generated = []
    loss_trajectories = []
    started = time.perf_counter()

    for batch_start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - batch_start)
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
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
            loss = loss_scale * per_sample_loss.sum()
            grad = torch.autograd.grad(loss, xt)[0]
            xt = xt - grad
            losses.append(per_sample_loss.detach().cpu())
            xt = model.sample_one_step(
                xt, ts[step], ts[step + 1], device, method="rk4"
            )

        generated.append(xt[:, 0].detach().cpu())
        loss_trajectories.append(torch.stack(losses, dim=1))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        torch.cat(generated, dim=0),
        torch.cat(loss_trajectories, dim=0),
        elapsed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--truth-cache", required=True)
    parser.add_argument("--truth-split", default="test", choices=["train", "test"])
    parser.add_argument("--truth-index", type=int, default=0)
    parser.add_argument("--reference-cache", nargs="+")
    parser.add_argument("--reference-max-samples", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--loss-scale", type=float, default=100.0)
    parser.add_argument("--subx", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_all(args.seed)

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

    model = get_flow_model(config.noise, config.fno, device)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint["model"])
    )
    model.eval()
    model.requires_grad_(False)

    truth = load_truth(
        args.truth_cache, args.truth_split, args.truth_index
    ).to(device)
    if tuple(truth.shape) != (resolution, resolution):
        raise ValueError(
            f"Truth shape {tuple(truth.shape)} does not match {resolution}x{resolution}"
        )

    generated, loss_trajectories, elapsed = sample_guided(
        model=model,
        truth=truth,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_eval=args.n_eval,
        loss_scale=args.loss_scale,
        subx=args.subx,
        device=device,
    )

    truth_cpu = truth.cpu()
    mean = generated.mean(dim=0)
    std = generated.std(dim=0, unbiased=False)
    error = mean - truth_cpu
    observed = torch.zeros_like(truth_cpu, dtype=torch.bool)
    observed[::args.subx, ::args.subx] = True
    unobserved = ~observed
    observed_sample_errors = (
        generated[:, observed] - truth_cpu[observed]
    ).square().mean(dim=1)

    spectrum_result = {}
    if args.reference_cache:
        reference = load_reference(
            args.reference_cache,
            args.truth_split,
            args.reference_max_samples,
        )
        k_values, generated_enstrophy, generated_energy = get_energy_spectrum(
            generated
        )
        reference_k, reference_enstrophy, reference_energy = get_energy_spectrum(
            reference
        )
        if not np.allclose(k_values, reference_k):
            raise RuntimeError("Generated and reference spectrum bins do not match")
        spectrum_result = {
            "reference_caches": args.reference_cache,
            "reference_n_samples": int(reference.shape[0]),
            "log_mse_enstrophy_spectrum": log_mse(
                generated_enstrophy, reference_enstrophy
            ),
            "log_mse_energy_spectrum": log_mse(
                generated_energy, reference_energy
            ),
            "k_values": np.asarray(k_values).tolist(),
            "generated_enstrophy_spectrum": np.asarray(
                generated_enstrophy
            ).tolist(),
            "reference_enstrophy_spectrum": np.asarray(
                reference_enstrophy
            ).tolist(),
            "generated_energy_spectrum": np.asarray(generated_energy).tolist(),
            "reference_energy_spectrum": np.asarray(reference_energy).tolist(),
        }

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "noise_type": str(config.noise.noise_type),
        "resolution": resolution,
        "fno_modes": modes,
        "truth_cache": args.truth_cache,
        "truth_split": args.truth_split,
        "truth_index": args.truth_index,
        "observation_fraction": float(observed.float().mean()),
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "n_eval": args.n_eval,
        "loss_scale": args.loss_scale,
        "seed": args.seed,
        "relative_l2_of_mean": float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(truth_cpu)
        ),
        "mse_of_mean": float(error.square().mean()),
        "observed_mse_of_mean": float(error[observed].square().mean()),
        "unobserved_mse_of_mean": float(error[unobserved].square().mean()),
        "observed_mse_per_generated_sample": float(
            observed_sample_errors.mean()
        ),
        "observed_mse_per_generated_sample_std": float(
            observed_sample_errors.std(unbiased=False)
        ),
        "average_posterior_std": float(std.mean()),
        "final_unscaled_observation_loss": float(
            loss_trajectories[:, -1].mean()
        ),
        "wall_seconds": elapsed,
        "seconds_per_sample": elapsed / args.n_samples,
    }
    result.update(spectrum_result)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(result, output_file, indent=2)
    if args.samples_output:
        os.makedirs(os.path.dirname(args.samples_output) or ".", exist_ok=True)
        torch.save(
            {
                "samples": generated,
                "truth": truth_cpu,
                "mean": mean,
                "std": std,
                "loss_trajectories": loss_trajectories,
                "result": result,
            },
            args.samples_output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
