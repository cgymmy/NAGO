#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if (( N_GPUS > 1 )); then
  echo 'The Allen-Cahn entry point supports single-process training; use N_GPUS=1.' >&2
  exit 2
fi
"$PYTHON_BIN" -m expt.ac.main --config configs/ac_configs.yaml --mode train \
  --noise-type "${SOURCE:-matern}" --savename "${RUN_NAME:-ac_${SOURCE:-matern}_fno}" \
  --data-file "${DATA_FILE:-data/ac/hwm10.pt}" "$@"
