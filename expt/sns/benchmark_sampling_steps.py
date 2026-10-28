"""Benchmark SNS generation accuracy and runtime versus ODE step count.

The script intentionally avoids the training entry point so that checkpoint
loading, dataset construction, and metric computation are excluded from the
reported sampling time.

Example:
  python expt/sns/benchmark_sampling_steps.py \
    --checkpoint gaussian=logs/sns/64_fno_mode64x64_gaussian/latest.pt \
    --steps 1 2 3 5 10 20 50 100 \
    --n-samples 1000 --repeats 3 \
    --output-dir logs/sns/speedup_res64_mode64/gaussian
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path


import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expt.sns.metrics import get_energy_spectrum
from expt.sns.main import get_flow_model, normalize_checkpoint_state_dict


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must have the form LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must have the form LABEL=PATH")
    return label, path


def log_mse(pred, target, eps=1e-12):
    pred = np.clip(np.asarray(pred), eps, None)
    target = np.clip(np.asarray(target), eps, None)
    return float(np.mean((np.log(pred) - np.log(target)) ** 2))


def evenly_spaced_indices(length, count):
    if count > length:
        raise ValueError(f"Cannot select {count} values from {length}")
    return torch.div(
        torch.arange(count, dtype=torch.int64) * length,
        count,
        rounding_mode="floor",
    )


def load_truth(
    processed_root,
    resolution,
    n_samples,
    num_files,
    frames_per_trajectory,
    reference_trajectories_per_file,
):
    """Load a trajectory-balanced held-out reference from the SNS caches."""
    per_file = (n_samples + num_files - 1) // num_files
    parts = []
    for file_index in range(1, num_files + 1):
        path = os.path.join(
            processed_root,
            f"data_file0{file_index}_lag0_hi{resolution}_split0p9_sub1.pt",
        )
        cached = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        test = cached["test"]
        if test.shape[0] % frames_per_trajectory:
            raise RuntimeError(
                f"{path} has {test.shape[0]} flattened test frames, which is "
                f"not divisible by {frames_per_trajectory}"
            )
        trajectories = test.reshape(
            -1, frames_per_trajectory, *test.shape[1:]
        )
        if trajectories.shape[0] < reference_trajectories_per_file:
            raise RuntimeError(
                f"{path} has only {trajectories.shape[0]} test trajectories"
            )
        reference = trajectories[:reference_trajectories_per_file]
        flat_reference = reference.reshape(-1, *test.shape[1:])
        indices = evenly_spaced_indices(flat_reference.shape[0], per_file)
        parts.append(flat_reference[indices, 0].clone())
        if sum(part.shape[0] for part in parts) >= n_samples:
            break
    truth = torch.cat(parts, dim=0)[:n_samples]
    if truth.shape[0] != n_samples:
        raise RuntimeError(
            f"Requested {n_samples} truth samples, found only {truth.shape[0]}"
        )
    return truth


def summarize(values):
    values = [float(value) for value in values]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def generate_samples(model, config, device, n_samples, batch_size, n_eval):
    parts = []
    for offset in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - offset)
        parts.append(
            model.sample(size, n_eval, config.sample_dims, device)[:, 0]
        )
    return torch.cat(parts, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=parse_checkpoint, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--n-truth",
        type=int,
        default=1000,
        help="Number of held-out reference samples used to estimate the target spectrum.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--processed-root", default="data/sns_processed"
    )
    parser.add_argument("--num-truth-files", type=int, default=5)
    parser.add_argument("--frames-per-trajectory", type=int, default=100)
    parser.add_argument(
        "--reference-trajectories-per-file",
        type=int,
        default=10,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    label, checkpoint_path = args.checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    checkpoint_step = int(checkpoint.get("step", -1))
    config = checkpoint["config"]
    resolution = int(config.sample_dims[-1])
    modes = [int(value) for value in config.fno.n_modes]
    if resolution != 64 or modes != [64, 64]:
        raise ValueError(
            f"Expected resolution 64 and modes [64, 64], got "
            f"resolution={resolution}, modes={modes}"
        )

    model_config = (
        config.fno
        if config.model_type == "fno"
        else getattr(config, config.model_type)
    )
    ratio = 4e-4 if config.noise.intp_type == "scale" else None
    model = get_flow_model(config.noise, model_config, device, ratio)
    model.load_state_dict(normalize_checkpoint_state_dict(checkpoint["model"]))
    model.eval()
    del checkpoint

    truth = load_truth(
        args.processed_root,
        resolution,
        args.n_truth,
        args.num_truth_files,
        args.frames_per_trajectory,
        args.reference_trajectories_per_file,
    )
    k_values, truth_enstrophy, truth_energy = get_energy_spectrum(truth)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with torch.no_grad():
        _ = model.sample(
            args.warmup_samples,
            max(args.steps),
            config.sample_dims,
            device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows = []
    spectra = {}
    for n_eval in args.steps:
        for repeat in range(args.repeats):
            # Reuse the same latent draws across step counts within a repeat.
            # This common-random-numbers design isolates integration-step
            # effects from finite-sample variation.
            run_seed = args.seed + repeat
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                generated_device = generate_samples(
                    model,
                    config,
                    device,
                    args.n_samples,
                    args.batch_size,
                    n_eval,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            materialization_start = time.perf_counter()
            generated = generated_device.cpu()
            materialization_seconds = (
                time.perf_counter() - materialization_start
            )
            end_to_end_seconds = elapsed + materialization_seconds

            _, generated_enstrophy, generated_energy = get_energy_spectrum(
                generated
            )
            row = {
                "method": label,
                "checkpoint": checkpoint_path,
                "checkpoint_step": checkpoint_step,
                "noise": str(config.noise.noise_type),
                "resolution": resolution,
                "modes": "x".join(map(str, modes)),
                "ode_steps": n_eval,
                "rk4_nfe": 4 * n_eval,
                "repeat": repeat,
                "seed": run_seed,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "elapsed_seconds": elapsed,
                "seconds_per_sample": elapsed / args.n_samples,
                "samples_per_second": args.n_samples / elapsed,
                "materialization_seconds": materialization_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "end_to_end_seconds_per_sample": (
                    end_to_end_seconds / args.n_samples
                ),
                "log_mse_enstrophy": log_mse(
                    generated_enstrophy, truth_enstrophy
                ),
                "log_mse_energy": log_mse(generated_energy, truth_energy),
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
            }
            rows.append(row)
            spectra[(n_eval, repeat)] = {
                "enstrophy": np.asarray(generated_enstrophy).tolist(),
                "energy": np.asarray(generated_energy).tolist(),
            }
            print(json.dumps(row), flush=True)
            del generated, generated_device

    raw_csv = output_dir / "sampling_steps_raw.csv"
    with raw_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for n_eval in args.steps:
        selected = [row for row in rows if row["ode_steps"] == n_eval]
        elapsed = summarize(row["elapsed_seconds"] for row in selected)
        materialization = summarize(
            row["materialization_seconds"] for row in selected
        )
        end_to_end = summarize(
            row["end_to_end_seconds"] for row in selected
        )
        energy = summarize(row["log_mse_energy"] for row in selected)
        enstrophy = summarize(row["log_mse_enstrophy"] for row in selected)
        summary_rows.append(
            {
                "method": label,
                "noise": str(config.noise.noise_type),
                "resolution": resolution,
                "modes": "x".join(map(str, modes)),
                "ode_steps": n_eval,
                "rk4_nfe": 4 * n_eval,
                "n_samples": args.n_samples,
                "repeats": args.repeats,
                "median_elapsed_seconds": elapsed["median"],
                "elapsed_iqr_seconds": elapsed["q3"] - elapsed["q1"],
                "median_seconds_per_sample": elapsed["median"] / args.n_samples,
                "median_samples_per_second": args.n_samples / elapsed["median"],
                "median_materialization_seconds": materialization["median"],
                "median_end_to_end_seconds": end_to_end["median"],
                "median_end_to_end_seconds_per_sample": (
                    end_to_end["median"] / args.n_samples
                ),
                "median_log_mse_energy": energy["median"],
                "energy_log_mse_iqr": energy["q3"] - energy["q1"],
                "median_log_mse_enstrophy": enstrophy["median"],
                "enstrophy_log_mse_iqr": enstrophy["q3"] - enstrophy["q1"],
            }
        )

    summary_csv = output_dir / "sampling_steps_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    result = {
        "method": label,
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "timing_scope": {
            "elapsed_seconds": (
                "CUDA-synchronized model sampling; output remains on device"
            ),
            "materialization_seconds": "device-to-host output transfer",
            "end_to_end_seconds": (
                "sampling plus device-to-host output transfer; excludes "
                "checkpoint loading, metric computation, and disk I/O"
            ),
        },
        "resolution": resolution,
        "modes": modes,
        "n_samples": args.n_samples,
        "n_truth": args.n_truth,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "steps": args.steps,
        "common_random_numbers_across_steps": True,
        "truth": {
            "processed_root": args.processed_root,
            "num_files": args.num_truth_files,
            "n_samples": args.n_truth,
            "frames_per_trajectory": args.frames_per_trajectory,
            "reference_trajectories_per_file": (
                args.reference_trajectories_per_file
            ),
            "k_values": np.asarray(k_values).tolist(),
            "enstrophy_spectrum": np.asarray(truth_enstrophy).tolist(),
            "energy_spectrum": np.asarray(truth_energy).tolist(),
        },
        "summary": summary_rows,
        "spectra": {
            f"steps_{step}_repeat_{repeat}": values
            for (step, repeat), values in spectra.items()
        },
    }
    with (output_dir / "sampling_steps_results.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
