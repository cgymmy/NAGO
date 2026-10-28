"""Distributed sampling and directional evaluation for anisotropic fields."""

import argparse
import json
import math
import os
import sys


import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from expt.anisotropic_matern.main import (
    build_flow,
    normalize_checkpoint_state_dict,
)
from expt.anisotropic_matern.metrics import (
    compare_generated_to_reference_power,
    compare_samples,
    normalized_power,
    theoretical_anisotropic_matern_power,
)
from src.random_field import AnisotropicMaternFieldGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument(
        "--reference",
        choices=("sampled", "theoretical"),
        default="sampled",
        help="Use Monte Carlo target samples or the exact discrete target spectrum.",
    )
    return parser.parse_args()


def setup_device():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return world_size, rank, local_rank, device


def generate_truth(config, count, resolution, seed, device, batch_size):
    target = config.target
    generator = AnisotropicMaternFieldGenerator(
        L=float(target.L),
        res=resolution - 1,
        nu=float(target.nu),
        ell_x=float(target.ell_x),
        ell_y=float(target.ell_y),
        theta=math.radians(float(target.theta_deg)),
        device=device,
    )
    parts = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        sample = generator.generate(
            size, seed=seed + start, normalize=False
        )[3]
        parts.append(sample.cpu())
    truth = torch.cat(parts)
    return ((truth - truth.mean()) / truth.std().clamp_min(1e-12)).numpy()




def main():
    args = parse_args()
    world_size, rank, _, device = setup_device()
    if args.n_samples % world_size:
        raise ValueError("n-samples must be divisible by world size.")

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    resolution = int(args.resolution or config.target.res)
    config.noise.res = resolution
    config.sample_dims[-2:] = [resolution, resolution]
    model = build_flow(config, device)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint["model"])
    )
    model.eval()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    local_count = args.n_samples // world_size
    generated_parts = []
    with torch.no_grad():
        for start in range(0, local_count, args.batch_size):
            size = min(args.batch_size, local_count - start)
            generated_parts.append(
                model.sample(
                    size,
                    args.n_eval,
                    [int(config.sample_dims[0]), resolution, resolution],
                    device,
                )[:, 0]
            )
    local_generated = torch.cat(generated_parts).contiguous()

    if world_size > 1:
        gathered = (
            [torch.empty_like(local_generated) for _ in range(world_size)]
            if rank == 0
            else None
        )
        dist.gather(local_generated, gather_list=gathered, dst=0)
        if rank != 0:
            dist.destroy_process_group()
            return
        generated = torch.cat(gathered).cpu().numpy()
    else:
        generated = local_generated.cpu().numpy()

    if args.reference == "theoretical":
        truth = None
        truth_power = theoretical_anisotropic_matern_power(
            resolution=resolution,
            domain_size=float(config.target.L),
            nu=float(config.target.nu),
            ell_x=float(config.target.ell_x),
            ell_y=float(config.target.ell_y),
            theta_deg=float(config.target.theta_deg),
        )
        metrics, arrays = compare_generated_to_reference_power(
            generated,
            truth_power,
            domain_size=float(config.target.L),
        )
    else:
        truth = generate_truth(
            config,
            args.n_samples,
            resolution,
            args.seed + 1_000_000,
            device,
            max(args.batch_size, 64),
        )
        metrics, arrays = compare_samples(
            generated, truth, domain_size=float(config.target.L)
        )
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint["step"]),
        "resolution": resolution,
        "n_samples": args.n_samples,
        "n_eval": args.n_eval,
        "world_size": world_size,
        "samples_per_rank": local_count,
        "reference_type": args.reference,
        "noise_type": str(config.noise.noise_type),
        "target": {
            "nu": float(config.target.nu),
            "ell_x": float(config.target.ell_x),
            "ell_y": float(config.target.ell_y),
            "theta_deg": float(config.target.theta_deg),
        },
        **metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    diagnostics = {"generated": generated, **arrays}
    if truth is not None:
        diagnostics["truth"] = truth
    np.savez_compressed(
        os.path.join(args.output_dir, "samples_and_diagnostics.npz"),
        **diagnostics,
    )
    print(json.dumps(result, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
