#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
launch -m expt.sns.main --config configs/sns_configs.yaml --mode train \
  --resolution 64 --fno-modes 64 64 --noise-type "${SOURCE:-matern}" \
  --matern-nu 0.5 --matern-ell 0.1 --global-batch-size 128 --num-workers 0 \
  --data-root "${DATA_ROOT:-data/sns}" \
  --processed-data-root "${DATA_CACHE:-data/sns_processed}" "$@"
