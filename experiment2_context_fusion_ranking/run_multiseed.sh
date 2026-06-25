#!/usr/bin/env bash
# Run the full experiment across several seeds, then aggregate to mean/std and a
# multiseed paired report so we can judge stability of any improvement.
set -euo pipefail

SEEDS=(1 2 3 4 5 42)
NBOOT=1000

for s in "${SEEDS[@]}"; do
    echo "=== seed ${s} ==="
    python context_fusion_ranking.py \
        --config configs/slice3d_context_fusion.yaml \
        --split_seed "${s}" \
        --output_dir "results/seed_${s}" \
        --n_bootstrap "${NBOOT}" \
        "$@"
done

python aggregate_multiseed.py --results_dir results
echo "Multiseed run complete -> results/multiseed_mean_std.csv, results/multiseed_paired_report.csv"
