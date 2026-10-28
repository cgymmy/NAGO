"""Metrics that retain directional information in two-dimensional fields."""

import math

import numpy as np
from scipy.ndimage import map_coordinates


def average_power_2d(samples):
    values = np.asarray(samples, dtype=np.float64)
    values = values - values.mean(axis=(-2, -1), keepdims=True)
    transformed = np.fft.fft2(values, axes=(-2, -1), norm="ortho")
    power = np.abs(transformed) ** 2
    return np.fft.fftshift(power.mean(axis=0))


def normalized_power(power, eps=1e-30):
    values = np.asarray(power, dtype=np.float64)
    return values / max(values.sum(), eps)


def spectral_ellipse(power):
    density = normalized_power(power)
    nx, ny = density.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0 / ny))
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    covariance = np.asarray(
        [
            [(density * Kx * Kx).sum(), (density * Kx * Ky).sum()],
            [(density * Kx * Ky).sum(), (density * Ky * Ky).sum()],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, 0]
    angle_deg = math.degrees(math.atan2(long_axis[1], long_axis[0])) % 180.0
    ratio = math.sqrt(
        max(float(eigenvalues[1]), 1e-30)
        / max(float(eigenvalues[0]), 1e-30)
    )
    return {
        "frequency_covariance": covariance.tolist(),
        "frequency_eigenvalues": eigenvalues.tolist(),
        "spatial_long_axis_angle_deg": angle_deg,
        "anisotropy_ratio": ratio,
    }


def angle_error_deg(first, second):
    difference = abs(float(first) - float(second)) % 180.0
    return min(difference, 180.0 - difference)


def log_power_mse(predicted, target, max_frequency_fraction=0.25, eps=1e-12):
    predicted = normalized_power(predicted)
    target = normalized_power(target)
    nx, ny = target.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0 / ny))
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    cutoff = max_frequency_fraction * min(nx, ny)
    mask = np.sqrt(Kx**2 + Ky**2) <= cutoff
    return float(
        np.mean(
            (
                np.log(np.clip(predicted[mask], eps, None))
                - np.log(np.clip(target[mask], eps, None))
            )
            ** 2
        )
    )


def relative_power_l1(predicted, target, eps=1e-12):
    predicted = normalized_power(predicted)
    target = normalized_power(target)
    return float(np.abs(predicted - target).sum() / max(np.abs(target).sum(), eps))


def autocorrelation_2d(power):
    unshifted = np.fft.ifftshift(np.asarray(power, dtype=np.float64))
    covariance = np.fft.ifft2(unshifted).real
    covariance = np.fft.fftshift(covariance)
    center = tuple(size // 2 for size in covariance.shape)
    return covariance / max(covariance[center], 1e-30)


def directional_correlation(correlation, angle_deg, domain_size=1.0):
    values = np.asarray(correlation, dtype=np.float64)
    nx, ny = values.shape
    count = min(nx, ny) // 2
    distances = np.linspace(0.0, domain_size / 2.0, count + 1)
    grid_steps = distances * min(nx, ny) / domain_size
    angle = math.radians(float(angle_deg))
    x_coords = nx // 2 + grid_steps * math.cos(angle)
    y_coords = ny // 2 + grid_steps * math.sin(angle)
    profile = map_coordinates(
        values,
        np.vstack([x_coords, y_coords]),
        order=1,
        mode="wrap",
    )
    return distances, profile


def e_folding_length(distances, correlation):
    distances = np.asarray(distances)
    correlation = np.asarray(correlation)
    threshold = math.exp(-1.0)
    below = np.flatnonzero(correlation <= threshold)
    if below.size == 0:
        return float(distances[-1])
    upper = int(below[0])
    if upper == 0:
        return float(distances[0])
    lower = upper - 1
    y0, y1 = correlation[lower], correlation[upper]
    if abs(y1 - y0) < 1e-15:
        return float(distances[upper])
    weight = (threshold - y0) / (y1 - y0)
    return float(distances[lower] + weight * (distances[upper] - distances[lower]))


def sample_moments(samples):
    values = np.asarray(samples, dtype=np.float64)
    mean = values.mean()
    centered = values - mean
    variance = np.mean(centered**2)
    return {
        "mean": float(mean),
        "variance": float(variance),
        "skewness": float(np.mean(centered**3) / max(variance, 1e-30) ** 1.5),
        "kurtosis": float(np.mean(centered**4) / max(variance, 1e-30) ** 2),
    }


def theoretical_anisotropic_matern_power(
    resolution,
    domain_size,
    nu,
    ell_x,
    ell_y,
    theta_deg,
):
    """Return the exact normalized-target power on the generator's FFT grid."""
    resolution = int(resolution)
    spacing = float(domain_size) / (resolution - 1)
    frequency = (
        np.fft.fftfreq(resolution, d=spacing) * (2.0 * math.pi)
    )
    Kx, Ky = np.meshgrid(frequency, frequency, indexing="ij")
    theta = math.radians(float(theta_deg))
    Kx_rot = math.cos(theta) * Kx + math.sin(theta) * Ky
    Ky_rot = -math.sin(theta) * Kx + math.cos(theta) * Ky
    ell_ref = math.sqrt(float(ell_x) * float(ell_y))
    scale_x = float(ell_x) / ell_ref
    scale_y = float(ell_y) / ell_ref
    kappa = math.sqrt(2.0 * float(nu)) / ell_ref
    alpha = float(nu) + 1.0
    inverse_filter = (
        kappa**2
        + (scale_x * Kx_rot) ** 2
        + (scale_y * Ky_rot) ** 2
    ) ** (-alpha)
    # average_power_2d removes the spatial mean of every field.
    inverse_filter[0, 0] = 0.0
    return np.fft.fftshift(inverse_filter)


def compare_generated_to_reference_power(
    generated,
    truth_power,
    domain_size=1.0,
    truth_moments=None,
):
    generated_power = average_power_2d(generated)
    truth_power = np.asarray(truth_power, dtype=np.float64)
    generated_ellipse = spectral_ellipse(generated_power)
    truth_ellipse = spectral_ellipse(truth_power)
    generated_correlation = autocorrelation_2d(generated_power)
    truth_correlation = autocorrelation_2d(truth_power)

    truth_angle = truth_ellipse["spatial_long_axis_angle_deg"]
    short_angle = (truth_angle + 90.0) % 180.0
    distance, generated_long = directional_correlation(
        generated_correlation, truth_angle, domain_size
    )
    _, truth_long = directional_correlation(
        truth_correlation, truth_angle, domain_size
    )
    _, generated_short = directional_correlation(
        generated_correlation, short_angle, domain_size
    )
    _, truth_short = directional_correlation(
        truth_correlation, short_angle, domain_size
    )

    metrics = {
        "log_power_2d_mse": log_power_mse(generated_power, truth_power),
        "relative_power_2d_l1": relative_power_l1(
            generated_power, truth_power
        ),
        "generated_spectral_ellipse": generated_ellipse,
        "truth_spectral_ellipse": truth_ellipse,
        "long_axis_angle_error_deg": angle_error_deg(
            generated_ellipse["spatial_long_axis_angle_deg"], truth_angle
        ),
        "anisotropy_ratio_absolute_error": abs(
            generated_ellipse["anisotropy_ratio"]
            - truth_ellipse["anisotropy_ratio"]
        ),
        "generated_moments": sample_moments(generated),
        "truth_moments": truth_moments
        or {
            "mean": 0.0,
            "variance": 1.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
        },
        "directional_e_folding_lengths": {
            "generated_long": e_folding_length(distance, generated_long),
            "truth_long": e_folding_length(distance, truth_long),
            "generated_short": e_folding_length(distance, generated_short),
            "truth_short": e_folding_length(distance, truth_short),
        },
    }
    arrays = {
        "generated_power": generated_power,
        "truth_power": truth_power,
        "generated_correlation": generated_correlation,
        "truth_correlation": truth_correlation,
        "distance": distance,
        "generated_long_correlation": generated_long,
        "truth_long_correlation": truth_long,
        "generated_short_correlation": generated_short,
        "truth_short_correlation": truth_short,
    }
    return metrics, arrays


def compare_samples(generated, truth, domain_size=1.0):
    return compare_generated_to_reference_power(
        generated,
        average_power_2d(truth),
        domain_size=domain_size,
        truth_moments=sample_moments(truth),
    )
