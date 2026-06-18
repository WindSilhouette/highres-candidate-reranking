#!/usr/bin/env bash
# Multi-seed robustness run: seeds 1..5, medium bootstrap / random budgets.
# Each seed writes its own results/seed_<N>/ folder; aggregate with
# aggregate_multiseed.py afterwards.
set -euo pipefail
cd "$(dirname "$0")"

DATA="data/lesions_embeddings.csv"

if [[ ! -f "$DATA" ]]; then
  echo "[run_multiseed] $DATA not found. Run: python prepare_embedding_csv.py" >&2
  exit 1
fi

for SEED in 1 2 3 4 5; do
  OUT="results/seed_${SEED}"
  mkdir -p "$OUT"
  echo "=================================================================="
  echo "[run_multiseed] seed ${SEED} -> ${OUT}"
  echo "=================================================================="
  python lesion_baseline.py \
    --use_synthetic_data false \
    --data_csv "$DATA" \
    --patient_id_col patient_id \
    --lesion_id_col lesion_id \
    --label_col malignant \
    --embedding_prefix emb_ \
    --context_method all \
    --split_seed "${SEED}" \
    --n_bootstrap 300 \
    --n_random_seeds 100 \
    --summary_csv "${OUT}/summary.csv" \
    --predictions_csv "${OUT}/predictions_best.csv" \
    2>&1 | tee "${OUT}/run.log"
done

echo "[run_multiseed] done. Now run: python aggregate_multiseed.py"
