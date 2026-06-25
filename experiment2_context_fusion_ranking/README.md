# Experiment 2 — Patient-Context Fusion Ranking (SLICE-3D)

Second-stage **patient-context lesion ranking** under a limited clinician review
budget. This experiment strengthens Experiment 1 and makes the comparison
**statistically defensible** with a paired patient-level bootstrap and a
multiseed stability check.

There is **no** high-resolution detection, YOLO, transformer, or segmentation
here. Lesion-level AUROC / AUPRC are reported but are **secondary**. The primary
metrics are patient-level top-k review-budget metrics.

## Research question

For each patient we rank that patient's lesions so the malignant lesion is
surfaced as early as possible. The question: **does fusing patient-context
features with the supervised risk score put the first malignant lesion earlier
in the review queue than classifier-only ranking — and is that gain real (CI
excludes 0) and stable across seeds?**

## How Experiment 2 differs from Experiment 1

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Methods | random, classifier, manual centroid/kNN fusion | adds `context_only`, `manual_fusion`, **`learned_fusion_logreg`**, optional `learned_fusion_mlp` |
| Context | centroid / kNN distance | full feature set: within-patient classifier rank/z-score/margin, centroid & cosine distance, kNN & max-NN distance, optional IsolationForest / LOF, lesion count, patient score stats |
| Fusion | fixed weighted sum | weighted sum **and** a learned model trained on TRAIN, selected on VAL |
| Statistics | per-method bootstrap CIs | **paired** patient bootstrap of every method vs `classifier_only`, plus a **multiseed** mean/std + stability report |
| `recall@k` | k ∈ {1,3,5} | k ∈ {1,3,5,10} |

## Data

Reuses the Experiment 1 prepared CSV:

```
data/lesions_embeddings.csv
columns: patient_id, lesion_id, malignant, emb_0 ... emb_2047
```

The CSV is **not** committed (see `.gitignore`). If it is absent you can still
smoke-test the whole pipeline with `--use_synthetic_data true`.

## Leakage controls (strict)

- Patient-disjoint train / val / test split; a hard assert checks no patient
  appears in two splits.
- The embedding scaler, the classifier, IsolationForest/LOF, the feature scaler,
  and the rank-normalisers are all **fit on TRAIN patients only**.
- The manual-fusion weight and the "best" method are selected on **VALIDATION**;
  final numbers come from **held-out TEST** patients.
- Context features for a test patient may use that patient's own lesion set
  (transductive patient-level ranking at inference) — but **test labels are never
  used** anywhere.
- All seeds are configurable (`--split_seed`).

## How to run

```bash
# fast smoke test (single seed, 50 bootstrap resamples)
bash run_debug.sh
# add synthetic data if the real CSV isn't present yet:
bash run_debug.sh --use_synthetic_data true

# full single-seed run (1000 bootstrap resamples)
bash run_full.sh

# all seeds (1 2 3 4 5 42) + aggregation
bash run_multiseed.sh
```

Direct CLI:

```bash
python context_fusion_ranking.py \
    --data_csv ../experiment1_contextual_lesion_prio/data/lesions_embeddings.csv \
    --patient_id_col patient_id --lesion_id_col lesion_id --label_col malignant \
    --embedding_prefix emb_ --split_seed 42 --output_dir results/seed_42 \
    --n_bootstrap 1000
```

## Outputs

Per seed, under `results/seed_<seed>/`:

- `summary.csv` — per-method TEST metrics (primary review-budget + secondary AUROC/AUPRC + review burden).
- `predictions_best.csv` — per-lesion test predictions for the best-on-validation method (`patient_id, lesion_id, true_label, classifier_score, context_score, fused_score, method, patient_rank`).
- `paired_comparison_report.csv` — paired patient-bootstrap deltas of each method vs `classifier_only` (long format: `method, metric, delta, ci_low, ci_high, ci_excludes_0, improves_over_baseline, n_malignant_patients`).
- `readable_report.txt` — plain-language verdict.

Across seeds, under `results/`:

- `multiseed_mean_std.csv` — mean/std of every metric per method.
- `multiseed_paired_report.csv` — per `(method, metric)`: mean/std delta, mean CI bounds, and the fraction of seeds where the CI excluded 0 in the better direction.

## How to interpret the output

1. In `summary.csv`, compare each method's `recall@5` and `mean_rank_first_malignant`
   against `classifier_only`. Lower mean rank / higher recall is better.
2. In `paired_comparison_report.csv`, a method genuinely helps on a metric when
   `improves_over_baseline = True` (its 95% CI excludes 0 in the better
   direction). A raw delta with a CI straddling 0 is **not** evidence.
3. In `multiseed_paired_report.csv`, look at `frac_seeds_improves`. A gain that
   holds on a majority of seeds is the one worth believing; a gain on a single
   seed is noise.
4. `readable_report.txt` states whether — at that seed — context/fusion improves
   over classifier-only, where, and whether the result supports continuing the
   patient-context ranking direction. The firm "continue vs rethink" call should
   be made from the **multiseed** report, not a single seed.

## Note on `context_only`

`context_only` ranks purely by the unsupervised centroid-deviation signal (no
supervised score). Experiment 1 already showed context alone does **not** replace
the classifier; it is included here as the honest lower-anchor for the fusion
methods.
