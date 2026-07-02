"""
src/feature_builder.py
======================
Patient-level retrieval features, in four groups:

  A. classifier features           (raw/calibrated score, within-patient rank/
                                     z-score, margins from patient mean/median/max)
  B. embedding context features    (centroid dist, cosine dist, kNN dists,
                                     density, within-patient outlier percentile)
  C. metadata features             (age, sex, site, tbp_lv_*, lesion counts)
  D. disagreement / hard-negative  (classifier-vs-context disagreement, gap from
                                     patient top score, same-site hard negative)

Leakage rule: groups B and C are label-free and seed-independent, so they are
built ONCE into features.parquet. Groups A and D depend on the first-stage
classifier, which must be fit on each seed's TRAIN patients only -- so they are
computed per seed at train/eval time via ``classifier_and_disagreement_features``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from src.io_utils import read_table, write_table

EPS = 1e-9
ID_COLS = ["isic_id", "patient_id", "malignant"]


# --------------------------------------------------------------------------- #
# Group B: embedding context (per patient, label-free)
# --------------------------------------------------------------------------- #
def compute_context_features(meta, E, k=5, min_lesions=2):
    n = len(meta)
    cols = {c: np.zeros(n) for c in
            ["centroid_eucl_dist", "cosine_centroid_dist", "knn_mean_dist",
             "knn_min_dist", "knn_max_dist", "local_density",
             "emb_outlier_pct_in_patient"]}
    idx = meta.index.to_numpy()
    for _, g in meta.groupby("patient_id"):
        rows = g.index.to_numpy()
        loc = np.searchsorted(idx, rows)
        X = np.asarray(E[loc], dtype=float)
        if len(rows) < min_lesions:
            continue
        c = X.mean(axis=0)
        d = np.linalg.norm(X - c, axis=1)
        cols["centroid_eucl_dist"][loc] = d
        cdir = c / (np.linalg.norm(c) + EPS)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        cols["cosine_centroid_dist"][loc] = 1.0 - Xn @ cdir
        kk = min(k, len(rows) - 1)
        # Patient groups are small enough that a direct distance matrix is faster
        # and avoids an unnecessary sklearn.neighbors dependency/import hit.
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        nd = np.sort(dist, axis=1)[:, :kk]
        cols["knn_mean_dist"][loc] = nd.mean(axis=1)
        cols["knn_min_dist"][loc] = nd.min(axis=1)
        cols["knn_max_dist"][loc] = nd.max(axis=1)
        cols["local_density"][loc] = 1.0 / (nd.mean(axis=1) + EPS)
        # within-patient percentile of centroid distance (0..1, 1 = most outlying)
        order = d.argsort().argsort()
        cols["emb_outlier_pct_in_patient"][loc] = order / max(len(rows) - 1, 1)
    return pd.DataFrame(cols, index=meta.index)


# --------------------------------------------------------------------------- #
# Group C: metadata features (deterministic encodings; numerics kept raw w/ NaN)
# --------------------------------------------------------------------------- #
def compute_metadata_features(meta):
    out = pd.DataFrame(index=meta.index)
    if "age_approx" in meta:
        out["age"] = pd.to_numeric(meta["age_approx"], errors="coerce")
    if "sex" in meta:
        out["sex_male"] = (meta["sex"].astype(str).str.lower() == "male").astype(float)
    if "anatom_site_general" in meta:
        site = meta["anatom_site_general"].astype(str).str.lower().fillna("unknown")
        for s in sorted(site.unique()):
            out[f"site_{s.replace('/', '_').replace(' ', '_')}"] = (site == s).astype(float)
        out["site_lesion_count"] = meta.groupby(
            ["patient_id", "anatom_site_general"])["isic_id"].transform("size").values
    for c in meta.columns:
        if c.startswith("tbp_lv_") or c == "clin_size_long_diam_mm":
            out[c] = pd.to_numeric(meta[c], errors="coerce")   # raw; imputed per-seed
    out["patient_lesion_count"] = meta.groupby("patient_id")["isic_id"].transform("size").values
    return out


# --------------------------------------------------------------------------- #
# Group A + D: classifier-derived + disagreement (PER SEED, fit on train only)
# --------------------------------------------------------------------------- #
def classifier_and_disagreement_features(meta, clf_score, context_df):
    """Compute groups A and D from a seed-specific classifier score.

    ``context_df`` supplies centroid distance for the disagreement features.
    Returns a DataFrame aligned to meta.index.
    """
    pid = meta["patient_id"].values
    s = pd.Series(clf_score, index=meta.index)
    g = s.groupby(meta["patient_id"])
    p_mean = g.transform("mean"); p_std = g.transform("std").fillna(0.0)
    p_med = g.transform("median"); p_max = g.transform("max")

    A = pd.DataFrame(index=meta.index)
    A["clf_score"] = clf_score
    A["clf_pct_rank_in_patient"] = g.rank(pct=True).values
    A["clf_zscore_in_patient"] = np.where(
        p_std.values > EPS, (clf_score - p_mean.values) / (p_std.values + EPS), 0.0)
    A["clf_margin_from_patient_mean"] = clf_score - p_mean.values
    A["clf_margin_from_patient_median"] = clf_score - p_med.values
    A["clf_margin_from_patient_max"] = clf_score - p_max.values

    # rank-normalise classifier and context to [0,1] for disagreement (global)
    def _rn(x):
        r = pd.Series(x).rank(pct=True).values
        return r
    clf_n = _rn(clf_score)
    ctx = context_df["centroid_eucl_dist"].values
    ctx_n = _rn(ctx)
    D = pd.DataFrame(index=meta.index)
    D["classifier_high_context_low"] = np.clip(clf_n - ctx_n, 0, None)
    D["context_high_classifier_low"] = np.clip(ctx_n - clf_n, 0, None)
    D["clf_context_disagreement"] = np.abs(clf_n - ctx_n)
    D["gap_from_top_score_in_patient"] = p_max.values - clf_score
    return pd.concat([A, D], axis=1)


# --------------------------------------------------------------------------- #
# Static build (groups B + C) -> features.parquet
# --------------------------------------------------------------------------- #
FEATURE_GROUPS_FILE = "feature_groups.json"


def build_static(cfg):
    from src.config import artifact_paths
    ap = artifact_paths(cfg)
    meta = read_table(ap["metadata"])
    E = np.load(ap["embeddings"], mmap_mode="r")

    ctx = compute_context_features(meta, E, k=5)
    md = compute_metadata_features(meta)
    feats = pd.concat([meta[ID_COLS], ctx, md], axis=1)
    write_table(feats, ap["features"], index=False)

    groups = {
        "B_context": list(ctx.columns),
        "C_metadata": list(md.columns),
    }
    with open(os.path.join(ap["work_dir"], FEATURE_GROUPS_FILE), "w") as f:
        json.dump(groups, f, indent=2)
    print(f">> wrote {ap['features']} with {feats.shape[1]} cols "
          f"(B={len(groups['B_context'])}, C={len(groups['C_metadata'])})")
    print(f">> feature groups manifest -> {FEATURE_GROUPS_FILE}\n")
    return feats, groups
