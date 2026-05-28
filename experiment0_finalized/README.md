# Experiment 0 — Second-Stage Contextual Reranking

> **Second-stage contextual reasoning for false-positive control after
> high-recall candidate generation in high-resolution lesion detection.**

---

## ⚠️ Toy Data Disclaimer

**Toy results are only pipeline validation, not evidence of clinical performance.**

All numbers produced by running this pipeline on synthetic data (`mode: toy`)
are generated from a controlled probability model and do not reflect:
- Real lesion appearance or morphology
- Clinical difficulty of melanoma detection
- Performance on actual patient populations
- Any regulatory or deployment claim

Before drawing any research conclusions, re-run the entire pipeline with
real SLICE-3D or iToBoS data using the instructions below.
Numbers from toy data should never appear in a paper or thesis result section.

---

## What This Experiment Tests

Given a patient with multiple lesion candidates (already detected),
does **contextual reranking** — using the relationship between lesions —
improve top-k sensitivity and review burden compared to a strong
calibrated independent classifier?

This is **not** the full detection experiment.
It isolates the second-stage reranking problem only.

---

## Project Structure

```
experiment0/
├── configs/
│   └── experiment0.yaml          # All hyperparameters
├── src/
│   ├── data/
│   │   ├── toy_generator.py      # Synthetic patient-grouped data
│   │   ├── dataset.py            # PyTorch dataset classes
│   │   ├── splitter.py           # Patient-disjoint splitting + leakage check
│   │   └── audit.py              # Data audit report
│   ├── models/
│   │   ├── classifier.py         # Independent lesion classifier (baseline)
│   │   └── calibration.py        # Temperature / Platt scaling (val only)
│   ├── rerankers/
│   │   ├── rerankers.py          # AbsoluteRisk, Centroid, kNN, TOAR-lite,
│   │   │                         # SetTransformer + score-flip sanity check
│   │   └── card.py               # CARD fusion (4 ablations)
│   ├── metrics/
│   │   ├── metrics.py            # SE@k, P@k, NNT, CandRed, AUROC + CIs
│   │   └── plots.py              # All output plots
│   └── training/
│       └── trainer.py            # Episode training + score-flip detection
├── scripts/
│   ├── verify_splits.py          # Standalone leakage checker
│   └── prepare_slice3d.py        # SLICE-3D CSV adapter
├── tests/
│   └── test_metrics.py           # 15 unit tests (pytest)
├── outputs/                      # All results written here
├── run_experiment0.py            # Main entry point
└── requirements.txt
```

---

## Quick Start (Toy Data)

```bash
pip install -r requirements.txt
python run_experiment0.py
```

---

## Run With Real SLICE-3D Data

```bash
# Step 1: Adapt your CSV
python scripts/prepare_slice3d.py \
    --input /data/slice3d/train-metadata.csv \
    --image-root /data/slice3d/train-image/ \
    --output outputs/slice3d_ready.csv

# Step 2: Run experiment
python run_experiment0.py --csv outputs/slice3d_ready.csv
```

---

## Verify No Patient Leakage (Run Anytime)

```bash
python scripts/verify_splits.py --csv outputs/toy_dataset.csv
```

---

## Run Unit Tests

```bash
python -m pytest tests/ -v
```

Expected: **15 passed**.

The tests include controlled examples where NNT@80 ≠ NNT@90,
proving the implementation is correct when sensitivity targets are
distinguishable (requires ≥ ~10 positive patients in the dataset).

---

## Rerankers Implemented

| Reranker | Type | Description |
|---|---|---|
| `absolute_risk` | Non-trainable | Calibrated classifier probability only |
| `centroid_euclidean` | Non-trainable | L2 distance from patient embedding mean |
| `centroid_cosine` | Non-trainable | 1 − cosine similarity to patient mean |
| `knn_k3` | Non-trainable | Mean dist to 3 nearest neighbours in patient |
| `knn_k5` | Non-trainable | Mean dist to 5 nearest neighbours in patient |
| `toar_lite` | Trainable | Population norm + residual deviation + MLP |
| `set_transformer` | Trainable | 2-layer multi-head self-attention over set |
| `card_abs_only` | Trainable | CARD: absolute risk signal only |
| `card_rel_only` | Trainable | CARD: relative anomaly signal only |
| `card_abs_rel` | Trainable | CARD: absolute + relative |
| `card_abs_rel_conflict` | Trainable | CARD: absolute + relative + conflict |

All unsupervised relative scorers have automatic score-direction
verification on the validation set — scores are flipped if AUROC < 0.5.

---

## Primary Metrics

All SE@k, MRR, and NNT metrics are computed on **positive patients only**
(patients with ≥ 1 malignant lesion). This is stated explicitly in all tables.

Precision@k and candidate_reduction@k5 are computed on **all patients**.

NNT@80 and NNT@90 are expected to be equal when fewer than ~10 positive
patients are available (achievable SE increments become too coarse).
The unit tests verify correct behaviour at both scales.

| Metric | Description | Patient scope |
|---|---|---|
| SE@k | Sensitivity at k reviewed | Positive patients |
| P@k | Precision at k reviewed | All patients |
| MRR | Mean reciprocal rank | Positive patients |
| NNT@80%sens | Reviews to reach 80% sensitivity | Positive patients |
| NNT@90%sens | Reviews to reach 90% sensitivity | Positive patients |
| CandRed@80/90 | Fraction of lesions not reviewed | Positive patients |
| AUROC | Area under ROC curve (secondary) | All lesions |
| ECE | Expected calibration error | Calibrated probs |

All primary metrics include 95% bootstrap confidence intervals
(500 bootstrap resamples over patients).

---

## Outputs

After running, `outputs/` contains:

```
metrics.json                    All metrics + CIs for every reranker
predictions.csv                 Per-lesion scores and ranks
calibration_report.json         ECE + Brier before/after calibration
split_report.txt                Patient counts + leakage proof
plots/
  se_at_k.png                   SE@k curve with 95% CI bands
  precision_at_k.png            Precision@k curve
  calibration_curve.png         Reliability diagram
  nnt_comparison.png            NNT bar chart per reranker
  score_distributions.png       Positive vs negative score histograms
  card_ablation_table.png       CARD ablation comparison table
```

---

## Known Limitations on Toy Data

1. **NNT@80 == NNT@90**: Expected with 4 test positive patients.
   Achievable SE = 0/0.25/0.50/0.75/1.0. Both 80% and 90% targets
   require SE=100%. This is correct behaviour, not a bug.
   Unit tests verify NNT@80 < NNT@90 when dataset is large enough.

2. **CARD rel_only and abs_rel perform poorly**: On toy data,
   the relative anomaly signal is weaker than on real images
   because embeddings are synthetic. Do not conclude from this
   that contextual reranking is useless.

3. **Small test set variance**: With 4 positive test patients,
   one correct/incorrect ranking changes SE@k by 0.25.
   Bootstrap CIs are wide as a result. This is expected.
