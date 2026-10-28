#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if (( $# < 1 )); then
  echo 'Usage: bash scripts/evaluate_matern.sh CHECKPOINT [evaluation options]' >&2
  exit 2
fi
checkpoint="$1"
shift
output_dir="${OUTPUT_DIR:-outputs/matern}"
mkdir -p "$output_dir"
launch -m expt.matern.test_spectral_distribution --checkpoint "$checkpoint" \
  --resolution 64 --n-eval 5 --n-samples 1000 --batch-size 32 \
  --samples-output "$output_dir/generated.pt" --truth-output "$output_dir/truth.pt" \
  --output "$output_dir/metrics.json" "$@"
