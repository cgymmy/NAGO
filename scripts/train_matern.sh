#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
launch -m expt.matern.main --config configs/matern_configs.yaml --mode train \
  --resolution 64 --fno-modes 64 64 --noise-type "${SOURCE:-matern}" \
  --nu 0.5 --ell 0.1 --num-workers 0 "$@"
