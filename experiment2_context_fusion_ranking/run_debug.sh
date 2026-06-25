#!/usr/bin/env bash
# Fast smoke test: single seed, low bootstrap. Good for verifying the pipeline
# end-to-end before a full run. Add `--use_synthetic_data true` if the real CSV
# is not present yet.
set -euo pipefail

python context_fusion_ranking.py \
    --config configs/slice3d_context_fusion.yaml \
    --split_seed 42 \
    --output_dir results/seed_42 \
    --n_bootstrap 50 \
    "$@"

echo "Debug run complete -> results/seed_42/"
