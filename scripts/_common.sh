#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_GPUS="${N_GPUS:-1}"
if [[ ! "$N_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'N_GPUS must be a positive integer.' >&2
  exit 2
fi

launch() {
  if (( N_GPUS > 1 )); then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$N_GPUS" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}
