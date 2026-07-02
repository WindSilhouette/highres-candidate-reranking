# FABLE-5 patched package notes

Patched for James's highres workflow.

## Fixes made

1. **Portable tabular I/O**
   - Added `src/io_utils.py`.
   - Uses parquet when `pyarrow`/`fastparquet` exists.
   - Falls back to pickle when parquet engines are unavailable, so local smoke tests do not die immediately.
   - Added `requirements.txt` with the expected Colab deps.

2. **Corrected error-report focus method**
   - `run_06_error_report.py` no longer always picks the first rank-aware method.
   - It uses the validation-selected best rank-aware method when possible.
   - If the overall validation winner is non-rank-aware, it picks the strongest rank-aware method from the result table only for visualization/error analysis.

3. **Replaced suspicious pAUC integration**
   - `src/evaluator.py` now uses an ISIC-style standardized partial AUC with `max_fpr = 1 - min_tpr` on flipped labels/scores.
   - This avoids the previous arbitrary slice integration.

4. **Generic dataset support for iToBoS**
   - `src/data_builder.py` now supports configurable/auto-detected `id_col`, `patient_col`, `target_col`, and `image_path_col`.
   - `src/embedding_extractor.py` now supports either ISIC HDF5 or a generic `image_dir`.
   - Added `configs/fable5_slice3d.yaml` and `configs/fable5_itobos.yaml` templates.

5. **Feature-builder speed/stability patch**
   - Removed unnecessary `sklearn.neighbors` dependency for within-patient kNN context features.
   - Uses direct numpy distance matrices per patient; this is faster and avoids import hangs in some sandboxes.

6. **Highres/Claude runner notebook**
   - Added `CLAUDE_FULL_RUN_SLICE3D_ITOBOS.ipynb`.
   - It locates the patched repo in a `highres` folder, writes separate configs, runs SLICE-3D and iToBoS, and prints final reports.

## Verification performed here

A synthetic smoke chain was run successfully through:

- `run_01_build_data.py`
- `run_02_extract_embeddings.py`
- `run_03_build_features.py`
- `run_04_train_eval.py --seed 42`
- `run_06_error_report.py --seed 42`

Because the sandbox does not contain the real SLICE-3D/iToBoS images, real-data execution must happen in Colab/highres.
