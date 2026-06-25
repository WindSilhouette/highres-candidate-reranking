#!/usr/bin/env bash
# Full single-seed run with 1000 bootstrap resamples on seed 42.
set -euo pipefail

python context_fusion_ranking.py \
    --config configs/slice3d_context_fusion.yaml \
    --split_seed 42 \
    --output_dir results/seed_42 \
    --n_bootstrap 1000 \
    "$@"

echo "Full run complete -> results/seed_42/"
