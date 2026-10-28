"""Generate SNS trajectory files accepted by the release training loader."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from gen_data.random_field import GaussianFieldGenerator
from gen_data.sns.generator_sns import navier_stokes_2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/sns'))
    parser.add_argument('--num-files', type=int, default=5)
    parser.add_argument('--trajectories-per-file', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--burn-in-time', type=float, default=50.0)
    parser.add_argument('--production-time', type=float, default=50.0)
    parser.add_argument('--record-interval', type=float, default=0.5)
    parser.add_argument('--viscosity', type=float, default=1e-3)
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--initial-alpha-q', type=float, default=0.002)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.num_files < 1 or args.trajectories_per_file < 1:
        parser.error('num-files and trajectories-per-file must be positive')
    if args.resolution < 4 or args.resolution % 2:
        parser.error('resolution must be even and >= 4')
    if args.production_time <= 0 or args.record_interval <= 0 or args.burn_in_time < 0:
        parser.error('production-time and record-interval must be positive; burn-in-time >= 0')
    # The original solver fixes its internal step at 1e-4.
    for label, interval in (('production-time', args.production_time),
                            ('burn-in-time', args.burn_in_time)):
        steps = interval / args.record_interval
        if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
            parser.error(f'{label} must be a multiple of record-interval')
    record_steps = args.record_interval / 1e-4
    if record_steps < 1 or not math.isclose(record_steps, round(record_steps), abs_tol=1e-8):
        parser.error('record-interval must be a positive multiple of the internal step 1e-4')
    paths = [args.output_dir / f'data_file0{i}.pt' for i in range(1, args.num_files + 1)]
    if any(p.exists() for p in paths):
        parser.error('Trajectory output already exists; choose a new output-dir')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, output in enumerate(paths):
        seed = args.seed + index
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = GaussianFieldGenerator(
            dim=2, L=2 * math.pi, res=args.resolution - 1,
            alpha=args.initial_alpha_q, device=device,
        )
        _, _, _, initial, _ = generator.generate(
            num_realizations=args.trajectories_per_file, seed=seed, normalize=True,
        )
        if args.burn_in_time:
            burn = navier_stokes_2d(
                [2 * math.pi] * 2, initial, args.viscosity, args.damping,
                args.burn_in_time, args.record_interval, sigma=1.0,
            )
            initial = burn[..., -1].contiguous()
            del burn
        solution = navier_stokes_2d(
            [2 * math.pi] * 2, initial, args.viscosity, args.damping,
            args.production_time, args.record_interval, sigma=1.0,
        )
        # Exclude the burn-in endpoint, then use [trajectory, time, x, y].
        fields = solution[..., 1:].permute(0, 3, 1, 2).contiguous().cpu()
        times = torch.arange(1, fields.shape[1] + 1) * args.record_interval
        times += args.burn_in_time
        if not torch.isfinite(fields).all():
            raise ValueError(f'Non-finite SNS fields in file {index}')
        torch.save((fields, times), output)
        print(f'Saved {tuple(fields.shape)} to {output}')
        del solution, fields
    metadata = vars(args).copy()
    metadata['output_dir'] = str(args.output_dir)
    metadata['internal_dt'] = 1e-4
    metadata['forcing'] = 'original eight-mode stochastic forcing, amplitude 1'
    (args.output_dir / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
