#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if (( $# != 2 )); then
  echo 'Usage: bash scripts/sweep_matern_steps.sh GAUSSIAN_CHECKPOINT MATERN_CHECKPOINT' >&2
  exit 2
fi
gaussian="$1"
matern="$2"
output_dir="${OUTPUT_DIR:-outputs/matern_step_sweep}"
read -r -a step_counts <<< "${STEPS:-2 3 4 5 8 10 15 20 30 50 75 100}"
for source in gaussian matern; do
  checkpoint="$gaussian"
  if [[ "$source" == matern ]]; then checkpoint="$matern"; fi
  for steps in "${step_counts[@]}"; do
    stem="$output_dir/$source/generation_eval_steps${steps}"
    mkdir -p "$(dirname "$stem")"
    launch -m expt.matern.test_spectral_distribution --checkpoint "$checkpoint" \
      --resolution "${RESOLUTION:-64}" --n-eval "$steps" \
      --n-samples "${N_SAMPLES:-1000}" --batch-size "${BATCH_SIZE:-32}" --seed 42 \
      --samples-output "${stem}_samples.pt" --truth-output "$output_dir/truth.pt" \
      --output "$stem.json"
  done
done
