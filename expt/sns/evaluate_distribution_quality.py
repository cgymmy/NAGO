"""Evaluate distributional and physical fidelity of 64x64 SNS samples.

The SNS state is interpreted as vorticity on the periodic domain [0, 2*pi)^2.
The script evaluates a trained NAGO checkpoint or a previously saved tensor
against held-out reference samples.  It intentionally complements (rather
than replaces) spectral log-MSE with distributional, non-Gaussian, multiscale,
and Navier--Stokes-specific diagnostics.

Checkpoint example (training server):

  python expt/sns/evaluate_distribution_quality.py \
    --checkpoint logs/sns/64_fno_mode64x64_matern_nu0.5_ell0.1/latest.pt \
    --processed-root data/sns_processed \
    --n-samples 1000 --n-eval 5 \
    --output-dir logs/sns/64_fno_mode64x64_matern_nu0.5_ell0.1/distribution_eval

Saved-sample example:

  python expt/sns/evaluate_distribution_quality.py \
    --truth-samples reference_64.pt --generated-samples nago_64.pt \
    --output-dir distribution_eval

Expected tensor shapes are (N,H,W) or (N,1,H,W).  For dictionary checkpoints,
the loader recognizes the keys ``samples``, ``generated``, ``truth``, and
``data``.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expt.sns.benchmark_sampling_steps import load_truth
from expt.sns.metrics import get_energy_spectrum
from expt.sns.main import get_flow_model, normalize_checkpoint_state_dict


EPS = 1e-12


def as_vorticity_tensor(value, label):
    """Extract a finite CPU float tensor with shape (N,H,W)."""
    if isinstance(value, dict):
        for key in ("samples", "generated", "truth", "data"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(
                f"{label} is a dictionary without a recognized sample key"
            )
    value = torch.as_tensor(value).detach().cpu()
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(
            f"{label} must have shape (N,H,W) or (N,1,H,W), got "
            f"{tuple(value.shape)}"
        )
    if value.shape[-2] != value.shape[-1]:
        raise ValueError(f"{label} must use a square grid")
    value = value.to(dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return value


def load_sample_tensor(path, label):
    value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return as_vorticity_tensor(value, label)


def evenly_spaced_subset(values, count):
    if count >= values.shape[0]:
        return values
    indices = torch.div(
        torch.arange(count, dtype=torch.int64) * values.shape[0],
        count,
        rounding_mode="floor",
    )
    return values[indices]


def generate_from_checkpoint(path, n_samples, n_eval, batch_size, device, seed):
    """Restore an SNS flow model and generate vorticity samples."""
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"{path} is not a complete SNS training checkpoint")
    config = checkpoint["config"]
    resolution = int(config.sample_dims[-1])
    if resolution != 64:
        raise ValueError(
            f"This evaluation is scoped to resolution 64, but the checkpoint "
            f"uses resolution {resolution}"
        )
    model_config = (
        config.fno
        if config.model_type == "fno"
        else getattr(config, config.model_type)
    )
    ratio = 4e-4 if config.noise.intp_type == "scale" else None
    model = get_flow_model(config.noise, model_config, device, ratio)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint["model"])
    )
    model.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parts = []
    with torch.no_grad():
        for offset in range(0, n_samples, batch_size):
            current = min(batch_size, n_samples - offset)
            generated = model.sample(
                current, n_eval, config.sample_dims, device
            )
            parts.append(generated[:, 0].detach().cpu())
    metadata = {
        "checkpoint": str(path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "noise": str(config.noise.noise_type),
        "model_type": str(config.model_type),
        "n_eval": int(n_eval),
    }
    if config.model_type == "fno":
        metadata["fno_modes"] = [
            int(value) for value in config.fno.n_modes
        ]
    return torch.cat(parts, dim=0), metadata


def velocity_from_vorticity(vorticity):
    """Recover incompressible velocity from periodic 2-D vorticity."""
    n = vorticity.shape[-1]
    dtype = vorticity.dtype
    device = vorticity.device
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype)
    kx = k[:, None]
    ky = k[None, :]
    k2 = kx.square() + ky.square()
    inverse_k2 = torch.where(k2 > 0, k2.reciprocal(), torch.zeros_like(k2))
    omega_hat = torch.fft.fft2(vorticity, norm="ortho")
    # A global sign depends on the vorticity/streamfunction convention and
    # does not affect the even-order or distributional diagnostics below.
    u_hat = 1j * ky * inverse_k2 * omega_hat
    v_hat = -1j * kx * inverse_k2 * omega_hat
    return torch.stack(
        [
            torch.fft.ifft2(u_hat, norm="ortho").real,
            torch.fft.ifft2(v_hat, norm="ortho").real,
        ],
        dim=1,
    )


def standardized_moments(values):
    values = values.to(dtype=torch.float64)
    mean = values.mean()
    centered = values - mean
    variance = centered.square().mean()
    std = variance.clamp_min(EPS).sqrt()
    return {
        "mean": float(mean),
        "standard_deviation": float(std),
        "skewness": float((centered.pow(3).mean() / std.pow(3))),
        "kurtosis": float((centered.pow(4).mean() / std.pow(4))),
    }


def quantile_wasserstein_1(first, second, max_values, quantiles, seed):
    """Approximate one-dimensional W1 using deterministic random subsamples."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def choose(values):
        values = values.detach().cpu().reshape(-1).to(torch.float64)
        if values.numel() > max_values:
            # Sampling with replacement avoids allocating a permutation as
            # large as the full collection of grid-point increments.
            indices = torch.randint(
                values.numel(), (max_values,), generator=generator
            )
            values = values[indices]
        return values

    first = choose(first)
    second = choose(second)
    probabilities = torch.linspace(0.0, 1.0, quantiles, dtype=torch.float64)
    first_q = torch.quantile(first, probabilities)
    second_q = torch.quantile(second, probabilities)
    return float(torch.trapezoid((first_q - second_q).abs(), probabilities))


def shared_histogram(first, second, bins=160, tail_probability=1e-4):
    """Return histograms on limits fixed by the first (reference) sample."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    lower, upper = np.quantile(
        first, [tail_probability, 1.0 - tail_probability]
    )
    if not np.isfinite(lower + upper) or upper <= lower:
        lower = float(np.min(first))
        upper = float(np.max(first) + EPS)
    edges = np.linspace(lower, upper, bins + 1)
    first_hist, _ = np.histogram(
        np.clip(first, lower, upper), bins=edges, density=False
    )
    second_hist, _ = np.histogram(
        np.clip(second, lower, upper), bins=edges, density=False
    )
    first_probability = first_hist.astype(np.float64)
    second_probability = second_hist.astype(np.float64)
    first_probability /= max(first_probability.sum(), 1.0)
    second_probability /= max(second_probability.sum(), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, first_probability, second_probability


def jensen_shannon_divergence(first_probability, second_probability):
    first = np.asarray(first_probability, dtype=np.float64) + EPS
    second = np.asarray(second_probability, dtype=np.float64) + EPS
    first /= first.sum()
    second /= second.sum()
    midpoint = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / midpoint))
        + 0.5 * np.sum(second * np.log(second / midpoint))
    )


def increment_values(field, separation):
    """Pool periodic x/y increments for scalar fields (N,H,W)."""
    dx = torch.roll(field, shifts=-separation, dims=-2) - field
    dy = torch.roll(field, shifts=-separation, dims=-1) - field
    return torch.cat([dx.reshape(-1), dy.reshape(-1)])


def longitudinal_velocity_increments(velocity, separation):
    """Pool longitudinal increments along the two coordinate directions."""
    along_x = (
        torch.roll(velocity[:, 0], shifts=-separation, dims=-2)
        - velocity[:, 0]
    )
    along_y = (
        torch.roll(velocity[:, 1], shifts=-separation, dims=-1)
        - velocity[:, 1]
    )
    return torch.cat([along_x.reshape(-1), along_y.reshape(-1)])


def structure_functions(velocity, separations):
    result = {2: [], 4: []}
    for separation in separations:
        increments = longitudinal_velocity_increments(velocity, separation)
        absolute = increments.abs().to(torch.float64)
        result[2].append(float(absolute.square().mean()))
        result[4].append(float(absolute.pow(4).mean()))
    return result


def relative_curve_error(generated, truth):
    generated = np.asarray(generated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(
        np.linalg.norm(generated - truth)
        / max(np.linalg.norm(truth), EPS)
    )


def spatial_correlation(field, vector=False):
    """Compute a normalized isotropic periodic two-point correlation."""
    if vector:
        centered = field - field.mean(dim=(-2, -1), keepdim=True)
        transformed = torch.fft.fft2(centered, norm="ortho")
        power = transformed.abs().square().sum(dim=1)
    else:
        centered = field - field.mean(dim=(-2, -1), keepdim=True)
        transformed = torch.fft.fft2(centered, norm="ortho")
        power = transformed.abs().square()
    correlation = torch.fft.ifft2(power, norm="ortho").real.mean(dim=0)
    correlation = torch.fft.fftshift(correlation)
    n = correlation.shape[-1]
    coordinate = torch.arange(n, dtype=torch.float64) - n // 2
    radius = torch.sqrt(
        coordinate[:, None].square() + coordinate[None, :].square()
    )
    bins = torch.floor(radius).to(torch.int64)
    max_radius = n // 2
    radial = []
    for value in range(max_radius + 1):
        selected = correlation[bins == value]
        radial.append(float(selected.mean()) if selected.numel() else math.nan)
    radial = np.asarray(radial, dtype=np.float64)
    if abs(radial[0]) > EPS:
        radial /= radial[0]
    return radial


def spectral_gradient(vorticity):
    n = vorticity.shape[-1]
    k = torch.fft.fftfreq(
        n, d=1.0 / n, device=vorticity.device, dtype=vorticity.dtype
    )
    transformed = torch.fft.fft2(vorticity, norm="ortho")
    gradient_x = torch.fft.ifft2(
        1j * k[:, None] * transformed, norm="ortho"
    ).real
    gradient_y = torch.fft.ifft2(
        1j * k[None, :] * transformed, norm="ortho"
    ).real
    return gradient_x, gradient_y


def physical_observables(vorticity, velocity):
    gradient_x, gradient_y = spectral_gradient(vorticity)
    centered = vorticity - vorticity.mean(dim=(-2, -1), keepdim=True)
    variance = centered.square().mean(dim=(-2, -1)).clamp_min(EPS)
    standard_deviation = variance.sqrt()
    return {
        "kinetic_energy": 0.5
        * velocity.square().sum(dim=1).mean(dim=(-2, -1)),
        "enstrophy": 0.5 * vorticity.square().mean(dim=(-2, -1)),
        "palinstrophy": 0.5
        * (gradient_x.square() + gradient_y.square()).mean(dim=(-2, -1)),
        "vorticity_skewness": centered.pow(3).mean(dim=(-2, -1))
        / standard_deviation.pow(3),
        "vorticity_kurtosis": centered.pow(4).mean(dim=(-2, -1))
        / variance.square(),
    }


def bootstrap_mean_difference(first, second, replicates, seed):
    """Independent-sample percentile CI for the generated-minus-truth mean."""
    first = first.detach().cpu().to(torch.float64)
    second = second.detach().cpu().to(torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    differences = []
    for _ in range(replicates):
        first_indices = torch.randint(
            first.numel(), (first.numel(),), generator=generator
        )
        second_indices = torch.randint(
            second.numel(), (second.numel(),), generator=generator
        )
        differences.append(
            float(first[first_indices].mean() - second[second_indices].mean())
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return [float(lower), float(upper)]


def multiscale_features(vorticity, truth_mean, truth_std):
    """Fixed multiscale physical-space representation for field-level MMD."""
    normalized = (vorticity - truth_mean) / max(truth_std, EPS)
    normalized = normalized[:, None]
    features = [
        F.adaptive_avg_pool2d(normalized, (size, size)).flatten(1)
        for size in (16, 8, 4)
    ]
    centered = normalized - normalized.mean(dim=(-2, -1), keepdim=True)
    scalar = torch.stack(
        [
            normalized.mean(dim=(-3, -2, -1)),
            normalized.square().mean(dim=(-3, -2, -1)).sqrt(),
            centered.pow(3).mean(dim=(-3, -2, -1)),
            centered.pow(4).mean(dim=(-3, -2, -1)),
        ],
        dim=1,
    )
    return torch.cat(features + [scalar], dim=1).to(torch.float64)


def rbf_gram(first, second, bandwidths):
    distances2 = torch.cdist(first, second).square()
    kernels = [
        torch.exp(-distances2 / (2.0 * bandwidth * bandwidth))
        for bandwidth in bandwidths
    ]
    return sum(kernels) / len(kernels)


def reference_kernel_bandwidths(reference, maximum_samples):
    """Fix MMD bandwidths using only the held-out reference distribution."""
    count = min(reference.shape[0], maximum_samples)
    reference = evenly_spaced_subset(reference, count)
    with torch.no_grad():
        distances = torch.pdist(reference)
        positive = distances[distances > 0]
        median = float(torch.median(positive)) if positive.numel() else 1.0
    return [max(median * factor, EPS) for factor in (0.5, 1.0, 2.0)]


def multiscale_mmd(first, second, maximum_samples, bandwidths=None):
    """Unbiased multi-bandwidth RBF MMD on fixed multiscale field features."""
    count = min(first.shape[0], second.shape[0], maximum_samples)
    first = evenly_spaced_subset(first, count)
    second = evenly_spaced_subset(second, count)
    pooled = torch.cat([first, second], dim=0)
    with torch.no_grad():
        if bandwidths is None:
            distances = torch.pdist(pooled)
            positive = distances[distances > 0]
            median = (
                float(torch.median(positive)) if positive.numel() else 1.0
            )
            bandwidths = [
                max(median * factor, EPS) for factor in (0.5, 1.0, 2.0)
            ]
        else:
            bandwidths = [float(value) for value in bandwidths]
        first_gram = rbf_gram(first, first, bandwidths)
        second_gram = rbf_gram(second, second, bandwidths)
        cross_gram = rbf_gram(first, second, bandwidths)
        diagonal_first = torch.diagonal(first_gram).sum()
        diagonal_second = torch.diagonal(second_gram).sum()
        denominator = count * (count - 1)
        if denominator <= 0:
            raise ValueError("MMD requires at least two samples per distribution")
        mmd2 = (
            (first_gram.sum() - diagonal_first) / denominator
            + (second_gram.sum() - diagonal_second) / denominator
            - 2.0 * cross_gram.mean()
        )
    return {
        "unbiased_mmd_squared": float(mmd2),
        "clipped_mmd": float(torch.clamp(mmd2, min=0).sqrt()),
        "sample_count_per_distribution": int(count),
        "kernel_bandwidths": bandwidths,
        "feature_dimension": int(first.shape[1]),
    }


def finite_sample_mmd_floor(truth_features, maximum_samples, bandwidths):
    count = min(truth_features.shape[0] // 2, maximum_samples)
    if count < 2:
        return None
    return multiscale_mmd(
        truth_features[:count],
        truth_features[-count:],
        count,
        bandwidths,
    )


def log_mse(first, second):
    first = np.clip(np.asarray(first, dtype=np.float64), EPS, None)
    second = np.clip(np.asarray(second, dtype=np.float64), EPS, None)
    return float(np.mean((np.log(first) - np.log(second)) ** 2))


def observable_summary(
    truth,
    generated,
    bootstrap_replicates,
    seed,
    max_wasserstein_values,
    wasserstein_quantiles,
):
    rows = {}
    for offset, name in enumerate(truth):
        truth_values = truth[name].detach().cpu().to(torch.float64)
        generated_values = generated[name].detach().cpu().to(torch.float64)
        truth_mean = float(truth_values.mean())
        generated_mean = float(generated_values.mean())
        truth_std = float(truth_values.std(unbiased=True))
        raw_w1 = quantile_wasserstein_1(
            truth_values,
            generated_values,
            max_wasserstein_values,
            wasserstein_quantiles,
            seed + offset,
        )
        absolute_mean_error = abs(generated_mean - truth_mean)
        relative_mean_error = (
            absolute_mean_error / abs(truth_mean)
            if abs(truth_mean) > 0.1 * max(truth_std, EPS)
            else None
        )
        rows[name] = {
            "truth_mean": truth_mean,
            "truth_standard_deviation": truth_std,
            "generated_mean": generated_mean,
            "generated_standard_deviation": float(
                generated_values.std(unbiased=True)
            ),
            "absolute_mean_error": absolute_mean_error,
            "standardized_mean_error": absolute_mean_error
            / max(truth_std, EPS),
            "relative_mean_error": relative_mean_error,
            "wasserstein_1": raw_w1,
            "standardized_wasserstein_1": raw_w1 / max(truth_std, EPS),
            "generated_minus_truth_mean_ci95": bootstrap_mean_difference(
                generated_values,
                truth_values,
                bootstrap_replicates,
                seed + 1000 + offset,
            ),
        }
    return rows


def flatten_numeric(prefix, value, rows):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, child, rows)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        rows.append({"metric": prefix, "value": value})


def save_curve_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "dataset",
                "statistic",
                "coordinate",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_curve_rows(rows, family, dataset, statistic, coordinates, values):
    for coordinate, value in zip(coordinates, values):
        rows.append(
            {
                "family": family,
                "dataset": dataset,
                "statistic": statistic,
                "coordinate": float(coordinate),
                "value": float(value),
            }
        )




def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate distributional and physical fidelity of 64x64 SNS samples"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--generated-samples")
    parser.add_argument("--truth-samples")
    parser.add_argument(
        "--processed-root", default="data/sns_processed"
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--num-truth-files", type=int, default=5)
    parser.add_argument("--frames-per-trajectory", type=int, default=100)
    parser.add_argument("--reference-trajectories-per-file", type=int, default=10)
    parser.add_argument(
        "--separations", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--mmd-max-samples", type=int, default=512)
    parser.add_argument("--max-wasserstein-values", type=int, default=1_000_000)
    parser.add_argument("--wasserstein-quantiles", type=int, default=2048)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--save-generated", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if any(value <= 0 or value > 32 for value in args.separations):
        parser.error("--separations must lie between 1 and 32")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}

    if args.generated_samples:
        generated = load_sample_tensor(
            args.generated_samples, "generated samples"
        )
        generated = evenly_spaced_subset(generated, args.n_samples)
        metadata["generated_samples_path"] = args.generated_samples
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but unavailable: {args.device}"
            )
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        generated, checkpoint_metadata = generate_from_checkpoint(
            args.checkpoint,
            args.n_samples,
            args.n_eval,
            args.batch_size,
            device,
            args.seed,
        )
        metadata.update(checkpoint_metadata)

    if args.truth_samples:
        truth = load_sample_tensor(args.truth_samples, "truth samples")
        truth = evenly_spaced_subset(truth, args.n_samples)
        metadata["truth_samples_path"] = args.truth_samples
    else:
        truth = load_truth(
            args.processed_root,
            64,
            args.n_samples,
            args.num_truth_files,
            args.frames_per_trajectory,
            args.reference_trajectories_per_file,
        )
        metadata["processed_root"] = args.processed_root

    generated = as_vorticity_tensor(generated, "generated samples")
    truth = as_vorticity_tensor(truth, "truth samples")
    count = min(generated.shape[0], truth.shape[0], args.n_samples)
    generated = evenly_spaced_subset(generated, count)
    truth = evenly_spaced_subset(truth, count)
    if generated.shape[-2:] != (64, 64) or truth.shape[-2:] != (64, 64):
        raise ValueError(
            f"Expected 64x64 samples, got truth={tuple(truth.shape[-2:])}, "
            f"generated={tuple(generated.shape[-2:])}"
        )

    truth_velocity = velocity_from_vorticity(truth)
    generated_velocity = velocity_from_vorticity(generated)
    truth_moments = standardized_moments(truth)
    generated_moments = standardized_moments(generated)

    truth_mean = float(truth.mean())
    truth_std = float(truth.std(unbiased=False))
    truth_features = multiscale_features(truth, truth_mean, truth_std)
    generated_features = multiscale_features(
        generated, truth_mean, truth_std
    )
    reference_bandwidths = reference_kernel_bandwidths(
        truth_features, args.mmd_max_samples
    )
    distribution_mmd = multiscale_mmd(
        truth_features,
        generated_features,
        args.mmd_max_samples,
        reference_bandwidths,
    )
    distribution_floor = finite_sample_mmd_floor(
        truth_features,
        args.mmd_max_samples,
        distribution_mmd["kernel_bandwidths"],
    )

    pointwise_w1 = quantile_wasserstein_1(
        truth,
        generated,
        args.max_wasserstein_values,
        args.wasserstein_quantiles,
        args.seed,
    )
    centers, truth_pdf, generated_pdf = shared_histogram(
        truth.numpy(), generated.numpy()
    )
    pointwise = {
        "wasserstein_1": pointwise_w1,
        "standardized_wasserstein_1": pointwise_w1 / max(truth_std, EPS),
        "jensen_shannon_divergence": jensen_shannon_divergence(
            truth_pdf, generated_pdf
        ),
        "truth_moments": truth_moments,
        "generated_moments": generated_moments,
        "moment_absolute_errors": {
            key: abs(generated_moments[key] - truth_moments[key])
            for key in truth_moments
        },
    }

    increment_metrics = {}
    for separation in args.separations:
        truth_increment = increment_values(truth, separation)
        generated_increment = increment_values(generated, separation)
        raw_w1 = quantile_wasserstein_1(
            truth_increment,
            generated_increment,
            args.max_wasserstein_values,
            args.wasserstein_quantiles,
            args.seed + separation,
        )
        truth_increment_std = float(truth_increment.std(unbiased=False))
        increment_centers, increment_truth_pdf, increment_generated_pdf = (
            shared_histogram(
                truth_increment.numpy(), generated_increment.numpy()
            )
        )
        increment_metrics[str(separation)] = {
            "wasserstein_1": raw_w1,
            "standardized_wasserstein_1": raw_w1
            / max(truth_increment_std, EPS),
            "jensen_shannon_divergence": jensen_shannon_divergence(
                increment_truth_pdf, increment_generated_pdf
            ),
            "truth_moments": standardized_moments(truth_increment),
            "generated_moments": standardized_moments(generated_increment),
        }
    truth_structure = structure_functions(
        truth_velocity, args.separations
    )
    generated_structure = structure_functions(
        generated_velocity, args.separations
    )
    truth_flatness = np.asarray(truth_structure[4]) / np.maximum(
        np.asarray(truth_structure[2]) ** 2, EPS
    )
    generated_flatness = np.asarray(generated_structure[4]) / np.maximum(
        np.asarray(generated_structure[2]) ** 2, EPS
    )
    structure_summary = {
        f"relative_l2_error_S{order}": relative_curve_error(
            generated_structure[order], truth_structure[order]
        )
        for order in (2, 4)
    }
    structure_summary["relative_l2_error_flatness_S4_over_S2_squared"] = (
        relative_curve_error(generated_flatness, truth_flatness)
    )

    truth_vorticity_correlation = spatial_correlation(truth)
    generated_vorticity_correlation = spatial_correlation(generated)
    truth_velocity_correlation = spatial_correlation(
        truth_velocity, vector=True
    )
    generated_velocity_correlation = spatial_correlation(
        generated_velocity, vector=True
    )
    correlation_summary = {
        "vorticity_relative_l2_error": relative_curve_error(
            generated_vorticity_correlation,
            truth_vorticity_correlation,
        ),
        "velocity_relative_l2_error": relative_curve_error(
            generated_velocity_correlation,
            truth_velocity_correlation,
        ),
    }

    truth_observables = physical_observables(truth, truth_velocity)
    generated_observables = physical_observables(
        generated, generated_velocity
    )
    observables = observable_summary(
        truth_observables,
        generated_observables,
        args.bootstrap_replicates,
        args.seed,
        args.max_wasserstein_values,
        args.wasserstein_quantiles,
    )

    k_values, truth_enstrophy_spectrum, truth_energy_spectrum = (
        get_energy_spectrum(truth)
    )
    _, generated_enstrophy_spectrum, generated_energy_spectrum = (
        get_energy_spectrum(generated)
    )
    spectral = {
        "log_mse_enstrophy": log_mse(
            generated_enstrophy_spectrum, truth_enstrophy_spectrum
        ),
        "log_mse_energy": log_mse(
            generated_energy_spectrum, truth_energy_spectrum
        ),
    }
    spectra = {
        "k": np.asarray(k_values),
        "truth_enstrophy": np.asarray(truth_enstrophy_spectrum),
        "generated_enstrophy": np.asarray(generated_enstrophy_spectrum),
        "truth_energy": np.asarray(truth_energy_spectrum),
        "generated_energy": np.asarray(generated_energy_spectrum),
    }

    result = {
        "protocol": {
            "state_variable": "vorticity",
            "domain": "[0, 2*pi)^2 periodic",
            "resolution": 64,
            "n_truth": int(truth.shape[0]),
            "n_generated": int(generated.shape[0]),
            "seed": args.seed,
            "separations_grid_points": args.separations,
            "structure_function_definition": (
                "isotropic longitudinal velocity increments, "
                "S_p(r)=E[|delta u_L(r)|^p]"
            ),
            "mmd_features": (
                "truth-normalized vorticity pooled to 16x16, 8x8, and 4x4, "
                "plus per-field moments"
            ),
            "mmd_bandwidth_selection": (
                "median heuristic on held-out truth features only; fixed "
                "across evaluated methods"
            ),
            "pdf_bin_selection": (
                "robust limits estimated from held-out truth only; fixed "
                "across evaluated methods"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "source": metadata,
        "overall_distribution_quality": {
            "multiscale_field_mmd": distribution_mmd,
            "truth_vs_truth_finite_sample_mmd_floor": distribution_floor,
        },
        "non_gaussian_structure": {
            "pointwise_vorticity": pointwise,
            "vorticity_increments_by_separation": increment_metrics,
        },
        "structure_functions": structure_summary,
        "spatial_correlations": correlation_summary,
        "sns_physical_observables": observables,
        "spectral_diagnostics": spectral,
        "artifacts": {
            "curve_csv": str(output_dir / "curves.csv"),
            "scalar_csv": str(output_dir / "scalar_metrics.csv"),
        },
    }

    curve_rows = []
    for dataset, structure, flatness in (
        ("truth", truth_structure, truth_flatness),
        ("generated", generated_structure, generated_flatness),
    ):
        append_curve_rows(
            curve_rows,
            "structure_function",
            dataset,
            "S2",
            args.separations,
            structure[2],
        )
        append_curve_rows(
            curve_rows,
            "structure_function",
            dataset,
            "S4",
            args.separations,
            structure[4],
        )
        append_curve_rows(
            curve_rows,
            "structure_function",
            dataset,
            "flatness_S4_over_S2_squared",
            args.separations,
            flatness,
        )
    correlation_radii = np.arange(len(truth_vorticity_correlation))
    for dataset, vorticity_correlation, velocity_correlation in (
        (
            "truth",
            truth_vorticity_correlation,
            truth_velocity_correlation,
        ),
        (
            "generated",
            generated_vorticity_correlation,
            generated_velocity_correlation,
        ),
    ):
        append_curve_rows(
            curve_rows,
            "spatial_correlation",
            dataset,
            "vorticity",
            correlation_radii,
            vorticity_correlation,
        )
        append_curve_rows(
            curve_rows,
            "spatial_correlation",
            dataset,
            "velocity",
            correlation_radii,
            velocity_correlation,
        )
    for dataset, enstrophy, energy in (
        (
            "truth",
            spectra["truth_enstrophy"],
            spectra["truth_energy"],
        ),
        (
            "generated",
            spectra["generated_enstrophy"],
            spectra["generated_energy"],
        ),
    ):
        append_curve_rows(
            curve_rows,
            "spectrum",
            dataset,
            "enstrophy",
            spectra["k"],
            enstrophy,
        )
        append_curve_rows(
            curve_rows,
            "spectrum",
            dataset,
            "energy",
            spectra["k"],
            energy,
        )
    save_curve_csv(output_dir / "curves.csv", curve_rows)
    scalar_rows = []
    flatten_numeric("", result, scalar_rows)
    with (output_dir / "scalar_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(scalar_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    if args.save_generated:
        generated_path = output_dir / "generated_samples.pt"
        torch.save(generated, generated_path)
        result["artifacts"]["generated_samples"] = str(generated_path)
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
