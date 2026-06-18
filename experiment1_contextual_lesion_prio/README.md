# Experiment 1 — Contextual Lesion Prioritization

Patient-context lesion prioritization for malignant-lesion triage under a limited
clinician review budget, evaluated on real lesion embeddings (SLICE-3D / ISIC 2024,
3D total-body photography).

This is a finalized, reproducible baseline experiment. It is a **packaging of the
V2.2 `lesion_baseline.py` runner** — no new architecture, no transformers, no plots.

---

## Research question

> Does patient-context lesion prioritization improve **top-k malignant lesion
> retrieval under a limited clinician review budget**, compared to classifier-only
> scoring, unsupervised context scoring alone, and random ranking?

The experiment evaluates melanoma screening as **patient-level lesion
prioritization**: for each patient, rank that patient's lesions so malignant
lesions appear as early as possible in the review list. It compares four scoring
strategies — classifier-only, unsupervised patient-context outlier scoring,
combined classifier+context, and random ranking — under top-k review budgets.

### Why this is not generic melanoma classification

Standard melanoma AI optimizes a context-naive, lesion-level binary classifier and
reports AUROC. That ignores the operational reality of screening: a clinician
reviews a *patient's* lesions under a time budget and wants the suspicious ones
surfaced first. This experiment therefore:

- ranks lesions **within each patient** (the "ugly duckling" framing: a malignant
  lesion stands out from that patient's own benign nevi), and
- makes the **primary metrics patient-level top-k / number-needed-to-review**, with
  AUROC/AUPRC kept only as secondary references.

The question is not "is this lesion malignant?" but "how few lesions must a
clinician review per patient before reaching the malignant one?"

### Primary metrics

- `recall@1`, `recall@3`, `recall@5` — fraction of malignant patients whose
  malignant lesion is within the top-K of their ranked list.
- `mean_rank_first_malignant` — average rank of the first malignant lesion.
- `mean_NNR_to_first_malignant` — number-needed-to-review (= mean rank of first
  malignant, by definition).
- `mean_percentile_rank_first_malignant` — lesion-count-normalized rank
  (rank ÷ patient lesion count), so a rank of 3 means different things for a
  patient with 5 vs. 150 lesions.

Secondary (references only): lesion-level `auroc`, `auprc`.

### Constraints (do not violate)

- Patient-disjoint splits only.
- Top-k metrics are primary; AUROC/AUPRC are secondary.
- No transformer / deep-learning architecture (this is the baseline stage).
- No plots yet.
- Do not change the research direction.

---

## Folder contents

```
experiment1_contextual_lesion_prioritization/
├── README.md                     # this file
├── lesion_baseline.py            # V2.2 runner (single-file, argparse)
├── prepare_embedding_csv.py      # metadata CSV + .npy  ->  per-lesion embedding CSV
├── run_debug.sh                  # fast smoke run (small budgets)
├── run_full.sh                   # main single-seed run (seed 42, full budgets)
├── run_multiseed.sh              # seeds 1..5 (medium budgets)
├── aggregate_multiseed.py        # aggregate per-seed test rows -> mean/std/count
├── configs/
│   └── slice3d_embedding_baseline.yaml   # config of record (scripts mirror it)
└── results/                      # all outputs land here (.gitkeep only in git)
```

---

## Required input format

`prepare_embedding_csv.py` expects two **row-aligned** inputs:

1. **Metadata CSV** (one row per lesion), with at least:
   - a patient id column (default `patient_id`),
   - a lesion id column (default `lesion_id`; falls back to `isic_id`, then to
     generated ids),
   - a label column (default `target`; falls back to `malignant` / `label`),
     either numeric `0/1` or strings `benign/malignant`, `true/false`, `yes/no`.
2. **Embeddings `.npy`** of shape `(n_lesions, D)`, where row `i` corresponds to
   metadata row `i` (same order).

It produces `data/lesions_embeddings.csv` with columns:

```
patient_id, lesion_id, malignant, emb_0, emb_1, ..., emb_{D-1}
```

This CSV is what every run script feeds to `lesion_baseline.py`.

---

## Setup

```bash
# Python 3.10+; from inside this folder:
pip install numpy pandas scikit-learn pyyaml
```

Place your prepared data where the defaults expect it (or pass explicit paths):

```
../data/processed/slice3d_subset.csv      # metadata
../data/processed/raw_embeddings.npy      # embeddings, row-aligned with metadata
```

---

## How to run

### 1. Prepare the embedding CSV

```bash
python prepare_embedding_csv.py
# or with explicit paths / columns:
python prepare_embedding_csv.py \
  --metadata ../data/processed/slice3d_subset.csv \
  --embeddings ../data/processed/raw_embeddings.npy \
  --output data/lesions_embeddings.csv \
  --patient_col patient_id --lesion_col lesion_id --label_col target
```

It prints row count, patient count, malignant lesions, malignant patients, and the
embedding dimension. Check those before running anything else.

### 2. Debug run (fast smoke test)

```bash
bash run_debug.sh
# -> results/debug/{summary.csv, predictions_best.csv, run.log}
```

### 3. Full single-seed run (the headline result)

```bash
bash run_full.sh
# -> results/seed_42/{summary.csv, predictions_best.csv, run.log}
```

### 4. Multi-seed robustness run + aggregation

```bash
bash run_multiseed.sh          # seeds 1..5 -> results/seed_1 ... results/seed_5
python aggregate_multiseed.py  # -> results/multiseed_all_test_rows.csv
                               #    results/multiseed_mean_std.csv
```

**Recommended order:** prepare → debug → full → multiseed → aggregate.

---

## What the outputs mean

Per run (`results/<run>/`):

- **`summary.csv`** — one row per `method × split` (val and test). Methods include
  `classifier_only`, `random`, and per context method `context_only[...]` and
  `combined[...]`. Columns are the primary metrics above plus
  `n_malignant_patients_evaluated`. **Read the `split == test` rows.**
- **`predictions_best.csv`** — per-lesion predictions for the best variant (chosen
  on validation `recall@3`) on the test split. Columns: ids, `true_label`,
  `classifier_score_raw/_norm`, `context_score_raw/_norm`, `combined_score`,
  `patient_rank` (1 = review first), `context_method`, `selected_weight`, `split`.
  Per-method `predictions_<method>.csv` are written alongside.
- **`run.log`** — full console output, including the patient-level bootstrap 95%
  CIs and the review-burden block.

Aggregated across seeds (`results/`):

- **`multiseed_all_test_rows.csv`** — every test-split row from every seed (long form).
- **`multiseed_mean_std.csv`** — per method, the mean / std / count of each primary
  metric across seeds. This is the table to report for stability.

---

## How to interpret success / failure

Look at the **test-split** rows (single seed) and the **mean ± std across seeds**.

**Success looks like:**

- `combined[...]` and/or `context_only[...]` clearly beat `random` on `recall@1/3/5`
  (random is the chance floor — with ~15–20 lesions/patient it sits near
  `recall@1 ≈ 0.06`, `recall@5 ≈ 0.30`).
- `mean_rank_first_malignant` / number-needed-to-review for the context or combined
  method is **substantially lower** than for `classifier_only` and `random`.
- The improvement is **stable across seeds** (small std in `multiseed_mean_std.csv`),
  and ideally supported by non-overlapping bootstrap CIs (see `run.log`).

This would support the thesis claim: patient-context prioritization surfaces
malignant lesions earlier under a tight review budget than a context-naive
classifier.

**Failure / null result looks like:**

- Context and combined methods are **no better than `classifier_only`**, or barely
  above `random` — i.e., the embeddings do not encode a usable per-patient
  "ugly duckling" signal at this stage.
- High variance across seeds (large std) — results are not yet trustworthy; revisit
  the embedding quality, the malignant-patient count, or the split sizes.

Either outcome is a legitimate, reportable baseline finding and motivates the
Phase 2 work (learned context scoring / set models). The point of this experiment
is an honest, reproducible measurement, not a guaranteed win.
