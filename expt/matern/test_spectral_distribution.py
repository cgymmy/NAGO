"""Generate samples from a Matérn checkpoint and compare radial power spectra."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from expt.matern.main import get_flow_model, normalize_checkpoint_state_dict
from models.fno import get_model as get_fno
from src.functional import GPPrior
from src.random_field import get_Fourier_spectrum


def log_mse(predicted, target, eps=1e-12):
    predicted = np.clip(np.asarray(predicted), eps, None)
    target = np.clip(np.asarray(target), eps, None)
    return float(np.mean((np.log(predicted) - np.log(target)) ** 2))


def relative_l1(predicted, target, eps=1e-12):
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    return float(
        np.mean(np.abs(predicted - target) / np.maximum(np.abs(target), eps))
    )


def sample_moments(samples):
    values = samples.double()
    mean = values.mean()
    centered = values - mean
    variance = centered.square().mean()
    return {
        'mean': mean.item(),
        'variance': variance.item(),
        'skewness': (
            centered.pow(3).mean() / variance.clamp_min(1e-30).pow(1.5)
        ).item(),
        'kurtosis': (
            centered.pow(4).mean() / variance.clamp_min(1e-30).square()
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-eval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples-output', required=True)
    parser.add_argument('--truth-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if args.n_samples % world_size != 0:
        raise ValueError(
            f'n_samples ({args.n_samples}) must be divisible by world size '
            f'({world_size}) for tensor gathering.'
        )
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group('nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    config = checkpoint['config']
    config.noise.res = args.resolution
    config.sample_dims = [
        int(config.sample_dims[0]), args.resolution, args.resolution
    ]
    if config.model_type != 'fno':
        raise ValueError('This resolution-grid test currently expects an FNO checkpoint.')

    model = get_flow_model(config.noise, get_fno(config.fno), device)
    model.load_state_dict(
        normalize_checkpoint_state_dict(checkpoint['model'])
    )
    model.eval()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    local_n = args.n_samples // world_size
    generated_parts = []
    with torch.no_grad():
        for start in range(0, local_n, args.batch_size):
            size = min(args.batch_size, local_n - start)
            generated_parts.append(
                model.sample(
                    size,
                    args.n_eval,
                    config.sample_dims,
                    device,
                    progress=rank == 0,
                    progress_desc=(
                        f'{os.path.basename(os.path.dirname(args.checkpoint))} '
                        f'{args.resolution}x{args.resolution}'
                    ),
                )[:, 0]
            )
    local_samples = torch.cat(generated_parts, dim=0).contiguous()

    if distributed:
        gathered = (
            [torch.empty_like(local_samples) for _ in range(world_size)]
            if rank == 0 else None
        )
        torch.distributed.gather(local_samples, gather_list=gathered, dst=0)
        if rank != 0:
            torch.distributed.destroy_process_group()
            return
        generated = torch.cat(gathered, dim=0).cpu()
    else:
        generated = local_samples.cpu()

    data_cfg = config.datasets
    if os.path.exists(args.truth_output):
        truth_item = torch.load(
            args.truth_output, map_location='cpu', weights_only=False
        )
        truth = truth_item['samples']
        if tuple(truth.shape) != (
            args.n_samples, args.resolution, args.resolution
        ):
            raise ValueError(
                f'Existing truth samples have shape {tuple(truth.shape)}, '
                f'expected {(args.n_samples, args.resolution, args.resolution)}.'
            )
    else:
        torch.manual_seed(args.seed)
        target_prior = GPPrior(
            dim=int(data_cfg.dim),
            L=float(data_cfg.L),
            res=args.resolution,
            nu=float(data_cfg.nu),
            ell=float(data_cfg.ell),
            device=device,
        )
        truth_parts = []
        truth_batch_size = max(args.batch_size, 64)
        for start in range(0, args.n_samples, truth_batch_size):
            size = min(truth_batch_size, args.n_samples - start)
            truth_parts.append(target_prior.sample(size, normalize=True).cpu())
        truth = torch.cat(truth_parts, dim=0)

    os.makedirs(os.path.dirname(args.samples_output) or '.', exist_ok=True)
    torch.save({
        'samples': generated,
        'checkpoint': args.checkpoint,
        'resolution': args.resolution,
        'n_eval': args.n_eval,
        'seed': args.seed,
    }, args.samples_output)
    if not os.path.exists(args.truth_output):
        os.makedirs(os.path.dirname(args.truth_output) or '.', exist_ok=True)
        torch.save({
            'samples': truth,
            'resolution': args.resolution,
            'nu': float(data_cfg.nu),
            'ell': float(data_cfg.ell),
            'seed': args.seed,
        }, args.truth_output)

    k_values, generated_spectrum = get_Fourier_spectrum(generated)
    truth_k_values, truth_spectrum = get_Fourier_spectrum(truth)
    if not np.allclose(k_values, truth_k_values):
        raise RuntimeError('Generated and truth spectrum bins do not match.')

    generated_moments = sample_moments(generated)
    truth_moments = sample_moments(truth)
    result = {
        'checkpoint': args.checkpoint,
        'samples': args.samples_output,
        'truth_samples': args.truth_output,
        'resolution': args.resolution,
        'n_samples': args.n_samples,
        'n_eval': args.n_eval,
        'world_size': world_size,
        'model_type': str(config.model_type),
        'fno_modes': [int(value) for value in config.fno.n_modes],
        'noise_type': str(config.noise.noise_type),
        'target_matern_nu': float(data_cfg.nu),
        'target_matern_ell': float(data_cfg.ell),
        'log_mse_power_spectrum': log_mse(
            generated_spectrum, truth_spectrum
        ),
        'mean_relative_power_spectrum_bias': relative_l1(
            generated_spectrum, truth_spectrum
        ),
        'generated_moments': generated_moments,
        'truth_moments': truth_moments,
        'moment_absolute_errors': {
            key: abs(generated_moments[key] - truth_moments[key])
            for key in truth_moments
        },
        'k_values': np.asarray(k_values).tolist(),
        'generated_power_spectrum': np.asarray(generated_spectrum).tolist(),
        'truth_power_spectrum': np.asarray(truth_spectrum).tolist(),
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as output_file:
        json.dump(result, output_file, indent=2)

    print(json.dumps({
        key: result[key] for key in (
            'resolution',
            'n_samples',
            'fno_modes',
            'log_mse_power_spectrum',
            'mean_relative_power_spectrum_bias',
        )
    }, indent=2))
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
