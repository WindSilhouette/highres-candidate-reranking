#!/usr/bin/env bash
# Main single-seed experiment (seed 42) with full bootstrap / random budgets.
set -euo pipefail
cd "$(dirname "$0")"

DATA="data/lesions_embeddings.csv"
OUT="results/seed_42"
mkdir -p "$OUT"

if [[ ! -f "$DATA" ]]; then
  echo "[run_full] $DATA not found. Run: python prepare_embedding_csv.py" >&2
  exit 1
fi

python lesion_baseline.py \
  --use_synthetic_data false \
  --data_csv "$DATA" \
  --patient_id_col patient_id \
  --lesion_id_col lesion_id \
  --label_col malignant \
  --embedding_prefix emb_ \
  --context_method all \
  --split_seed 42 \
  --n_bootstrap 1000 \
  --n_random_seeds 200 \
  --summary_csv "$OUT/summary.csv" \
  --predictions_csv "$OUT/predictions_best.csv" \
  2>&1 | tee "$OUT/run.log"

echo "[run_full] done -> $OUT/{summary.csv, predictions_best.csv, run.log}"
