#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
launch -m expt.anisotropic_matern.main --resolution 64 --fno-modes 64 64 \
  --noise-type "${SOURCE:-matern}" --noise-nu 0.5 --noise-ell 0.05 \
  --batch-size 128 --amp fp32 "$@"
