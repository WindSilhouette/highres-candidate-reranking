#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_embedding_csv.py
=================================================================================
Build the per-lesion embedding CSV that lesion_baseline.py consumes, from:
    * a metadata CSV (one row per lesion: patient id, lesion id, label, ...)
    * a 2-D embedding .npy of shape (n_lesions, D), ROW-ALIGNED with the metadata

Output columns (exactly):
    patient_id, lesion_id, malignant, emb_0, emb_1, ..., emb_{D-1}

Defaults match the experiment layout:
    --metadata    ../data/processed/slice3d_subset.csv
    --embeddings  ../data/processed/raw_embeddings.npy
    --output      data/lesions_embeddings.csv

Column resolution:
    patient_id : --patient_col (default 'patient_id'); must exist.
    lesion_id  : --lesion_col (default 'lesion_id'); if missing, fall back to
                 'isic_id'; if that is also missing, generate unique ids.
    malignant  : --label_col (default 'target'); if missing, try 'malignant',
                 'target', 'label'. Mapped to 0/1 (numeric 0/1 or common
                 strings benign/malignant, true/false, yes/no).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

LABEL_MAP = {
    "0": 0, "1": 1,
    "benign": 0, "malignant": 1,
    "false": 0, "true": 1,
    "no": 0, "yes": 1,
}


def map_label(s: pd.Series, source_name: str) -> np.ndarray:
    """Map a label column to 0/1 with no silent coercion."""
    if s.isna().any():
        sys.exit(f"[FATAL] label column '{source_name}' has missing/blank values.")
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy()
        if not np.isin(v, [0, 1]).all():
            bad = sorted(set(np.unique(v).tolist()) - {0, 1, 0.0, 1.0})
            sys.exit(f"[FATAL] label column '{source_name}' has non-binary numeric "
                     f"values {bad}; expected only 0/1.")
        return v.astype(int)
    norm = s.astype(str).str.strip().str.lower()
    bad = sorted(set(norm.unique()) - set(LABEL_MAP))
    if bad:
        sys.exit(f"[FATAL] label column '{source_name}' has unmappable values {bad}. "
                 f"Supported (case-insensitive): {sorted(LABEL_MAP)}.")
    return norm.map(LABEL_MAP).to_numpy().astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the per-lesion embedding CSV "
                                             "for lesion_baseline.py.")
    ap.add_argument("--metadata", default="../data/processed/slice3d_subset.csv")
    ap.add_argument("--embeddings", default="../data/processed/raw_embeddings.npy")
    ap.add_argument("--output", default="data/lesions_embeddings.csv")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--lesion_col", default="lesion_id")
    ap.add_argument("--label_col", default="target")
    ap.add_argument("--emb_prefix", default="emb_")
    a = ap.parse_args()

    if not os.path.exists(a.metadata):
        sys.exit(f"[FATAL] metadata CSV not found: {a.metadata}")
    if not os.path.exists(a.embeddings):
        sys.exit(f"[FATAL] embeddings .npy not found: {a.embeddings}")

    meta = pd.read_csv(a.metadata)
    emb = np.load(a.embeddings)
    if emb.ndim != 2:
        sys.exit(f"[FATAL] embeddings array must be 2-D (n_lesions, D); got shape {emb.shape}.")
    if len(meta) != emb.shape[0]:
        sys.exit(f"[FATAL] metadata rows ({len(meta)}) != embedding rows ({emb.shape[0]}). "
                 "The metadata CSV and the .npy must be row-aligned (same order).")

    # --- patient id ---------------------------------------------------------
    if a.patient_col not in meta.columns:
        sys.exit(f"[FATAL] patient column '{a.patient_col}' not in metadata columns: "
                 f"{list(meta.columns)[:12]}...")
    patient_id = meta[a.patient_col].astype(str).to_numpy()

    # --- lesion id (with fallbacks) ----------------------------------------
    if a.lesion_col in meta.columns:
        lesion_id = meta[a.lesion_col].astype(str).to_numpy()
        lesion_src = a.lesion_col
    elif "isic_id" in meta.columns:
        lesion_id = meta["isic_id"].astype(str).to_numpy()
        lesion_src = "isic_id (fallback)"
    else:
        lesion_id = np.array([f"L{i:06d}" for i in range(len(meta))])
        lesion_src = "generated (fallback)"

    # --- label (with fallbacks) --------------------------------------------
    label_candidates = [a.label_col, "malignant", "target", "label"]
    lab_col = next((c for c in label_candidates if c in meta.columns), None)
    if lab_col is None:
        sys.exit(f"[FATAL] no label column found; tried {label_candidates}.")
    malignant = map_label(meta[lab_col], lab_col)

    # --- assemble + write ---------------------------------------------------
    D = emb.shape[1]
    emb_block = {f"{a.emb_prefix}{i}": emb[:, i] for i in range(D)}
    out = pd.DataFrame({"patient_id": patient_id,
                        "lesion_id": lesion_id,
                        "malignant": malignant,
                        **emb_block})

    out_dir = os.path.dirname(os.path.abspath(a.output))
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(a.output, index=False)

    # --- report -------------------------------------------------------------
    n_rows = len(out)
    n_patients = out["patient_id"].nunique()
    n_mal = int((malignant == 1).sum())
    n_mal_patients = out.loc[malignant == 1, "patient_id"].nunique()
    print("=" * 70)
    print("prepare_embedding_csv.py")
    print("=" * 70)
    print(f"  metadata        : {a.metadata}")
    print(f"  embeddings      : {a.embeddings}  (shape {emb.shape})")
    print(f"  lesion_id source: {lesion_src}")
    print(f"  label source    : '{lab_col}' -> written as 'malignant'")
    print(f"  output          : {a.output}")
    print("-" * 70)
    print(f"  rows (lesions)        : {n_rows}")
    print(f"  unique patients       : {n_patients}")
    print(f"  malignant lesions     : {n_mal}")
    print(f"  malignant patients    : {n_mal_patients}")
    print(f"  embedding dimension D : {D}")
    print("=" * 70)
    if n_mal_patients < 10:
        print("  !  CAUTION: <10 malignant patients — top-k metrics and bootstrap CIs "
              "will be unstable. Consider a larger subset.")


if __name__ == "__main__":
    main()
