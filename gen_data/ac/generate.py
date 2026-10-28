"""Write Allen--Cahn MCMC fields in the format consumed by expt/ac/main.py."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gen_data.ac.experiments_allen_cahn import benchmark_samplers_spde


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--samples-per-chain', type=int, default=10000)
    parser.add_argument('--burn-in', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('data/ac/hwm10.pt'))
    args = parser.parse_args()
    if args.grid_size < 4 or args.samples_per_chain < 2 or args.burn_in < 0:
        parser.error('grid-size >= 4, samples-per-chain >= 2, and burn-in >= 0 are required')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}; choose a new output path')

    # The existing sampler uses N+1 grid points and sets its own seed to 42.
    results, _, spacing = benchmark_samplers_spde(
        N=args.grid_size - 1,
        n_samples=args.samples_per_chain,
        burn_in=args.burn_in,
    )
    result = results['Hamiltonian Walk Move n=10']
    samples = np.asarray(result['samples'])
    if samples.ndim != 2 or samples.shape[1] != args.grid_size:
        raise ValueError(f'Unexpected field shape: {samples.shape}')
    if not np.isfinite(samples).all():
        raise ValueError('The sampler returned non-finite fields')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(samples).float(), args.output)
    metadata = {
        'sampler': 'Hamiltonian Walk Move n=10',
        'seed': 42,
        'grid_size': args.grid_size,
        'spacing': spacing,
        'samples_per_chain': args.samples_per_chain,
        'burn_in': args.burn_in,
        'saved_shape': list(samples.shape),
        'mean_acceptance_rate': float(np.mean(result['acceptance_rates'])),
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {samples.shape[0]} fields to {args.output}')


if __name__ == '__main__':
    main()
