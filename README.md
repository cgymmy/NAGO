# NAGO: Noise-Aware Generative Operator via Flow Matching

Code for **NAGO**, accepted as a poster at **NeurIPS 2026 Main Track**.
NAGO designs the source noise distribution for functional flow matching to
sample stochastic fields and invariant measures of SPDEs.

This directory is a standalone code release. It contains the method,
training and numerical-test entry points, data generators, configurations,
and launch scripts. Datasets and pretrained weights are not distributed;
training creates checkpoints locally.

For the method insight, benchmark figures, and guidance comparison, see
[Theory and results](RESULTS.md).

## Repository structure

```text
NAGO/
├── README.md
├── RESULTS.md                    # Theory, experiments, and figure gallery
├── LICENSE                      # MIT for original NAGO contributions
├── THIRD_PARTY_NOTICES.md        # Third-party terms and attribution
├── CITATION.cff
├── licenses/                    # Upstream license texts and notice
├── requirements.txt
├── src/                         # Flow matching, noise priors, random fields, utilities
├── models/                      # Time-conditioned FNO and U-Net (1D/2D)
├── configs/                     # Matérn, Allen–Cahn, and SNS training configurations
├── expt/
│   ├── matern/                  # Isotropic fields, spectra, resolution and guidance tests
│   ├── ac/                      # Stochastic Allen–Cahn training and sampling
│   ├── sns/                     # SNS training, distribution metrics and timing
│   └── anisotropic_matern/      # Directional-field experiment, configuration and metrics
├── gen_data/
│   ├── ac/                      # Ensemble MCMC and training-data export
│   └── sns/                     # Pseudo-spectral solver and trajectory export
├── scripts/                     # Portable generation, training, evaluation and sweep commands
└── assets/                      # Small, verified result illustrations
```

`expt/` contains training, numerical testing, and the metric functions they
require. Tests export JSON/CSV metrics and optional sample tensors. Standalone
plotting, result aggregation, and presentation scripts are not included.

Run commands from the repository root. The shell scripts also locate this directory themselves.
Runtime directories `data/`, `logs/`, and `outputs/` are created as needed and
are excluded from Git, together with checkpoint and sample files.

## Installation

```bash
git clone https://github.com/cgymmy/NAGO.git
cd NAGO
```

The experiment environment uses **Python 3.9**, **PyTorch 2.6.0**,
**torchvision 0.21.0**, **CUDA 12.4**, and **neuraloperator 0.3.0**.
GPU training uses CUDA; distributed training uses NCCL.

Create and activate a Python 3.9 environment, then install PyTorch for your
platform. For the CUDA 12.4 environment:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The core dependency versions are taken from the experiment environment.
A fresh-environment installation and complete paper reproduction are still
to be validated.

## Quick start: Matérn fields

The Matérn experiment generates its target fields procedurally, so it needs
no downloaded dataset. The launch script uses a 64×64 target, a 64×64-mode
FNO, and a Matérn source with `nu=0.5`, `ell=0.1`.

```bash
# Train the Matérn-source model on one GPU.
bash scripts/train_matern.sh

# Train the matched Gaussian-source baseline.
SOURCE=gaussian bash scripts/train_matern.sh

# Evaluate a checkpoint produced by training.
bash scripts/evaluate_matern.sh \
  logs/matern/64_fno_mode64x64_matern_nu0.5_ell0.1/latest.pt
```

Evaluation writes `outputs/matern/metrics.json`, generated fields, and
reference fields. It computes spectra and moments without rendering figures. These files are runtime outputs, not bundled
assets. No checkpoint is supplied with this release.

Use `SOURCE=white` for the white-noise ablation. To override resolutions,
source parameters, or the configuration, pass the underlying CLI options:

```bash
bash scripts/train_matern.sh --resolution 128 --fno-modes 32 32 \
  --nu 0.5 --ell 0.05
OUTPUT_DIR=outputs/matern_128 bash scripts/evaluate_matern.sh \
  logs/matern/128_fno_mode32x32_matern_nu0.5_ell0.05/latest.pt \
  --resolution 128 --n-eval 10
```

## Other benchmarks

### Stochastic Allen–Cahn

Generate a 64-point field dataset using the existing Hamiltonian walk move
sampler, then train. The data generator uses seed 42 and ten leapfrog steps.
`samples-per-chain` counts samples after burn-in; the exporter flattens all
chains into the training tensor.

```bash
bash scripts/generate_ac.sh --grid-size 64 \
  --samples-per-chain 10000 --burn-in 1000
bash scripts/train_ac.sh
SOURCE=gaussian bash scripts/train_ac.sh

# Sample from a checkpoint trained above.
python -m expt.ac.main --mode inf --config configs/ac_configs.yaml \
  --resume logs/ac/ac_matern_fno/latest.pt \
  --data-file data/ac/hwm10.pt --n-eval 5 \
  --samples-output outputs/ac/generated.pt
```

The trainer accepts a tensor of shape `[samples, grid_size]` in a `.pt` file.
Use `DATA_FILE=/path/to/fields.pt bash scripts/train_ac.sh` for existing data.
This entry point supports single-process training.

### Stochastic Navier–Stokes (SNS)

The generator calls the existing solver and exports five trajectory files
named `data_file01.pt` through `data_file05.pt`. Each contains `(fields, times)`,
with fields of shape `[trajectories, snapshots, x, y]`.

```bash
bash scripts/generate_sns.sh --resolution 256 \
  --num-files 5 --trajectories-per-file 10 \
  --burn-in-time 50 --production-time 50 --record-interval 0.5
bash scripts/train_sns.sh
SOURCE=gaussian bash scripts/train_sns.sh
bash scripts/evaluate_sns.sh \
  logs/sns/64_fno_mode64x64_matern_nu0.5_ell0.1/latest.pt
```

The solver has an internal time step of `1e-4`; full trajectory generation
is substantially more expensive than the neural sampler. The trainer
resizes fields to its configured resolution and saves reusable processed
data in `data/sns_processed/`. It preserves the original normalization
constant `3.0679163932800293`.

To use existing data and caches:

```bash
DATA_ROOT=/path/to/SNS DATA_CACHE=/path/to/SNS_processed \
  bash scripts/train_sns.sh
```

Freshly generated datasets are supported by the loader; their identity to
the original paper datasets has not been established. Dataset size, burn-in,
sampling correlations, normalization, and seeds matter when comparing results.

### Stationary anisotropic Matérn target

The target has `nu=2`, `ell_x=0.1`, `ell_y=0.05`, and a 30-degree rotation.
The launch scripts use the full 64×64 grid of Fourier modes and compare
Gaussian and isotropic Matérn sources.

```bash
bash scripts/train_anisotropic.sh
SOURCE=gaussian bash scripts/train_anisotropic.sh
bash scripts/evaluate_anisotropic.sh \
  logs/anisotropic_matern/EXPERIMENT_NAME/latest.pt
```

Use the experiment directory printed by training in place of
`EXPERIMENT_NAME`. The single-GPU example uses global batch 128 and FP32.
The paper experiment used global batch 3072 and FP16; select that setting with:

```bash
N_GPUS=8 bash scripts/train_anisotropic.sh --batch-size 3072 --amp fp16
```

Evaluation uses 1024 samples and ten RK4 steps by default.

## Running scripts

| Setting | Purpose |
|---|---|
| `PYTHON_BIN` | Python executable from the active environment; default `python` |
| `N_GPUS` | Process count for supported distributed training/evaluation; default `1` |
| `SOURCE` | Source distribution for training: `matern`, `gaussian`, or supported `white` |
| `DATA_ROOT`, `DATA_CACHE` | SNS raw trajectory and processed-cache directories |
| `DATA_FILE` | Allen–Cahn tensor path |
| `OUTPUT_DIR` | Evaluation/sweep output directory |

For example, `N_GPUS=8 bash scripts/train_matern.sh` starts DDP training.
Keep the global batch size and, for distributed evaluation, the sample count
divisible by the process count. Matérn, SNS, and anisotropic training support
DDP; SNS metric evaluation runs in one process.

Training saves checkpoints under `logs/<benchmark>/<experiment>/` and records
TensorBoard summaries. For options, use the module's help, for example:

```bash
python -m expt.matern.main --help
python -m expt.sns.evaluate_distribution_quality --help
tensorboard --logdir logs
```

The experiment modules and `scripts/` provide RK4 sweeps, cross-resolution
tests, guidance controls, distribution metrics, and timing. Use each module's
`--help` for its arguments.


## License and citation

Original NAGO contributions are distributed under the [MIT License](LICENSE).
Third-party components retain their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.md), including MIT and Apache 2.0
components.

Author metadata is provided in [CITATION.cff](CITATION.cff).
