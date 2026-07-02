# FABLE-5 — Budget-Aware Melanoma Lesion Retrieval

**F**ull-data **A**blation for **B**udget-aware **L**esion **E**valuation.

A patient-level retrieval benchmark for melanoma screening. Instead of scoring
lesions independently, FABLE-5 asks a sharper clinical question:

> **Can rank-aware, patient-level training move malignant lesions earlier in the
> within-patient review list than ordinary classifier-only scoring?**

Each patient's lesions are ranked against each other, and we measure how many
lesions a clinician must review before reaching the malignant one. AUROC/AUPRC
are reported but **secondary**; the primary metrics are review-budget metrics
(recall@k, first-malignant rank, number-needed-to-review, normalized percentile).

---

## Why this exists (relation to Experiments 1–3)

Experiments 1–2 on the real ISIC 2024 / SLICE-3D subset showed that patient
context carries signal — manual context fusion lifted recall@5 (~0.60 → ~0.65)
and recall@10 (~0.72 → ~0.81) and shaved ~1 lesion off the mean first-malignant
rank — but ordinary *lesion-level* learned fusion collapsed back to the
classifier and bootstrap CIs still crossed zero. The diagnosis was an **objective
mismatch**: lesion-level binary cross-entropy is not within-patient top-k ranking.

FABLE-5 is the research engine that tests the resulting hypothesis directly: it
optimizes the **patient-level ranking objective** (pairwise / listwise / lambda),
mines hard same-patient negatives, and evaluates strictly on review-budget
metrics with a paired patient-level bootstrap. It is built to be a reusable
platform, not a one-off notebook, so a positive result can seed a paper.

It deliberately does **not** build a YOLO/iToBoS detector or an SSL foundation
model yet — those are the *pivots* the conclusion logic recommends only if
rank-aware retrieval fails.

---

## Structure

```
fable5_budget_aware_retrieval/
  configs/                 smoke / medium / full run configs
  src/
    config.py              config loader + RUN_MODE presets + artifact paths
    data_builder.py        metadata load/audit + patient-disjoint split files
    embedding_extractor.py ResNet50/EffNet-B0 over HDF5 (lazy torch) + synthetic fallback
    feature_builder.py     feature groups A/B/C/D (leakage-safe split)
    rank_models.py         all 10 models + hard-negative mining + numpy rankers
    evaluator.py           review-budget metrics + AUROC/AUPRC/pAUC
    bootstrap.py           paired patient-level bootstrap
    error_analysis.py      case CSVs, stratified analysis, curves, plots
    report_writer.py       per-seed report + readable_report.md (A/B/C verdict)
  run_01_build_data.py     -> artifacts/metadata.parquet + splits/
  run_02_extract_embeddings.py -> artifacts/embeddings.npy (+ emb_index.json)
  run_03_build_features.py -> artifacts/features.parquet (groups B + C)
  run_04_train_eval.py     one seed: train, eval, bootstrap -> results/seed_<s>/
  run_05_multiseed.py      all seeds + aggregate + readable_report.md
  run_06_error_report.py   error CSVs + plots
  FABLE5_Colab_Runbook.ipynb
  results/.gitkeep
```

Data and results are **never committed** (see `.gitignore`). Artifacts are stored
as parquet (metadata/features), `.npy`/memmap (embeddings), and JSON (splits) —
never as giant 2048-column CSVs.

---

## Patched highres workflow

This patched version adds generic image-folder support for iToBoS-style datasets in addition to the original SLICE-3D/ISIC HDF5 path. Use:

- `configs/fable5_slice3d.yaml` for SLICE-3D / ISIC 2024 metadata + HDF5.
- `configs/fable5_itobos.yaml` for iToBoS / generic metadata CSV + image folder.
- `CLAUDE_FULL_RUN_SLICE3D_ITOBOS.ipynb` to run both datasets from a `highres` folder in Colab/Drive.

The core scripts now support `paths.image_dir` and `data.{id_col, patient_col, target_col, image_path_col}`. If column names are standard, the loader auto-detects them; otherwise set the names in the YAML or notebook.

`src/io_utils.py` keeps local tests runnable without parquet engines by falling back to pickle, but Colab should still install `pyarrow` from `requirements.txt` for proper parquet files.

---

## Run modes

| mode   | lesions      | use |
|--------|--------------|-----|
| SMOKE  | ~12k         | fast end-to-end sanity of the whole engine |
| MEDIUM | 50k–75k      | realistic power for the paired bootstrap |
| FULL   | all feasible | final numbers (falls back to MEDIUM if Colab OOMs) |

Set the real data paths in a config (or via the notebook flags):

```yaml
paths:
  metadata_csv: /content/isic2024/train-metadata.csv
  hdf5:         /content/isic2024/train-image.hdf5
```

If `metadata_csv` / `hdf5` are left `null`, the engine generates
**synthetic ISIC-shaped data with a complementary within-patient signal**, so the
full pipeline (and every model) runs and can be validated without Kaggle.

### Quick start (synthetic, local)

```bash
python run_01_build_data.py       --config configs/fable5_smoke.yaml
python run_02_extract_embeddings.py --config configs/fable5_smoke.yaml
python run_03_build_features.py   --config configs/fable5_smoke.yaml
python run_05_multiseed.py        --config configs/fable5_smoke.yaml
python run_06_error_report.py     --config configs/fable5_smoke.yaml --seed 42
```

---

## Models compared

`random`, `classifier_only`, `metadata_model`, `context_only`,
`manual_fusion_validation_selected`, `pointwise_logreg_fusion` (the Exp-2 failure
baseline), `pairwise_rank_logreg`, `pairwise_rank_mlp`, `listwise_softmax_ranker`,
and `lambda_pairwise_logreg`.

The rank-aware trainers are pure numpy (siamese pairwise MLP, ListNet-style
listwise softmax, LambdaRank-style pair weighting), so the core runs fast on CPU
with no torch dependency. torch/torchvision/h5py are needed **only** by
`embedding_extractor` on the real image path and are imported lazily.

### Feature groups

- **A — classifier**: raw score, within-patient percentile & z-score, margins from
  patient mean/median/max.
- **B — embedding context**: centroid Euclidean/cosine distance, within-patient
  kNN distances, local density, within-patient outlier percentile.
- **C — metadata**: age, sex, anatomical site, `tbp_lv_*` fields, patient &
  site lesion counts.
- **D — disagreement / hard-negative**: classifier-vs-context disagreement, gap
  from the patient's top score, same-site hard-negative signal.

Groups B and C are label-free and seed-independent, so they are built once into
`features.parquet`. Groups A and D depend on the first-stage classifier, which is
**fit on each seed's train patients only**, so they are computed per seed at
train/eval time. All imputation/scaling is fit on train.

---

## How to interpret the output

`results/readable_report.md` answers, in plain language: what data was used, what
baselines were compared, whether rank-aware beat classifier-only and on which
top-k metrics, whether it was significant, where it helped/failed, and the next
scientific decision. The paired patient-level bootstrap (resampling patients, same
resample for both methods) reports Δ and 95% CI for recall@1/3/5/10/20, mean rank,
NNR, and normalized percentile, and flags whether each CI excludes 0.

**Conclusion logic (printed and written to the report):**

- **A — strong success**: a rank-aware method improves recall@5 or recall@10 over
  classifier_only with the CI excluding 0 on a majority of seeds **and** improves
  mean first-malignant rank. Strengthen and write up.
- **B — promising but not significant**: point estimates improve but CIs cross 0.
  Add statistical power (MEDIUM/FULL, more malignant patients, pooled test) and
  richer/calibrated features before scaling the model.
- **C — negative**: rank-aware does not beat classifier_only. Pivot to
  representation learning / SSL embeddings or high-resolution detection.

---

## Leakage controls

- Patient-disjoint train/val/test splits, written once per seed and shared by
  every model (identical patients for all comparisons).
- The first-stage classifier and all classifier-derived features are fit on train
  patients only; hard-negative pairs are mined only within train patients.
- Imputers and scalers are fit on train and applied to val/test.
- Hyper-parameters and the reported "best" method are selected on **validation**
  (primary recall@5, then mean first-malignant rank, then recall@10), never test.
