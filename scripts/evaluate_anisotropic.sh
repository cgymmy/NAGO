#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if (( $# < 1 )); then
  echo 'Usage: bash scripts/evaluate_anisotropic.sh CHECKPOINT [evaluation options]' >&2
  exit 2
fi
checkpoint="$1"
shift
launch -m expt.anisotropic_matern.evaluate --checkpoint "$checkpoint" \
  --n-samples 1024 --n-eval 10 --output-dir "${OUTPUT_DIR:-outputs/anisotropic}" "$@"
