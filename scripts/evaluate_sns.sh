#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if (( $# < 1 )); then
  echo 'Usage: bash scripts/evaluate_sns.sh CHECKPOINT [evaluation options]' >&2
  exit 2
fi
checkpoint="$1"
shift
output_dir="${OUTPUT_DIR:-outputs/sns}"
mkdir -p "$output_dir"
"$PYTHON_BIN" -m expt.sns.test_spectral_bias --checkpoint "$checkpoint" \
  --data-root "${DATA_ROOT:-data/sns}" \
  --processed-data-root "${DATA_CACHE:-data/sns_processed}" \
  --n-eval 5 --n-samples 1000 --gen-batch-size 32 \
  --output "$output_dir/metrics.json" "$@"
