#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lesion_baseline.py  —  Version 2.2  (hardening pass over V2.1)
=================================================================================
Patient-level melanoma lesion-triage baseline (SLICE-3D / ISIC 2024 style data).

The task this script scores:
    1. From per-lesion embeddings, learn (a) a simple supervised malignant
       classifier and (b) unsupervised per-patient "ugly-duckling" context
       scores (how much a lesion deviates from the patient's own lesion set).
    2. Combine them and RANK each patient's lesions so the lesions a clinician
       should review first are surfaced under a small review budget (top-k).

PRIMARY metric is patient-level top-k recall / number-needed-to-review, NOT
AUROC. Lesion-level AUROC / AUPRC are reported only as secondary references.

This is a non-parametric / linear baseline scaffold on purpose. No transformer,
no deep architecture, no plotting. Those are later phases.

V2.2 hardening (safety / evaluation only — no redesign):
    * strict label validation (numeric 0/1 or common string mappings; else stop)
    * single-class training split no longer crashes the classifier
    * score normalization is fit on TRAIN ONLY and applied to val/test
      (no transductive use of val/test distributions); train context scores
      are computed to fit the context normalizers
    * real-data hygiene: NaN-embedding stop/drop, duplicate-id and <2-lesion warns
    * predictions files carry both raw and normalized scores
    * review burden printed for classifier_only, best combined, and each method

QUICK START
    python lesion_baseline.py                        # synthetic demo, all methods
    python lesion_baseline.py --help                 # all flags
    (real-data command is printed at the end of every run)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================ #
# CONFIG  — defaults; every entry is overridable from the command line.        #
# ============================================================================ #
CONFIG = {
    # --- data ---------------------------------------------------------------
    "data_csv": "lesions.csv",          # used when use_synthetic_data is False
    "use_synthetic_data": True,         # set False to load data_csv instead
    "patient_id_col": "patient_id",
    "lesion_id_col": "lesion_id",
    "label_col": "malignant",           # 0 / 1 (or mappable strings)
    "embedding_prefix": "emb_",         # every column starting with this = a feature
    "allow_drop_nan": False,            # real data: drop NaN-embedding rows instead of stopping

    # --- split (patient-disjoint, stratified by malignant patient) ----------
    "val_frac": 0.20,                   # fraction of PATIENTS held out for val
    "test_frac": 0.20,                  # fraction of PATIENTS held out for test
    "split_seed": 42,

    # --- classifier ---------------------------------------------------------
    "classifier": "logreg",             # "logreg" (default) or "mlp"
    "mlp_hidden": (64,),

    # --- context (unsupervised per-patient outlier) score -------------------
    "context_method": "all",            # "all" | "centroid" | "cosine_centroid" | "knn"
    "context_k": 5,                     # neighbours for the knn method
    "min_lesions_for_context": 2,       # patients below this get context score 0.0

    # --- score fusion -------------------------------------------------------
    # combined = w * classifier_norm + (1 - w) * context_norm   (w = clf weight)
    # Normalizers are fit on TRAIN ONLY (min-max), then applied to val/test.
    # w is selected on VALIDATION by recall@3 (tie-break: lower mean rank of
    # first malignant). Final top-k ranking is always WITHIN patient.
    "weight_grid": (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    "select_metric": "recall@3",

    # --- evaluation ---------------------------------------------------------
    "k_list": (1, 3, 5),
    "n_bootstrap": 1000,                # patient-level bootstrap iterations
    "n_random_seeds": 200,              # random-ranking baseline averaging
    "ci_alpha": 0.05,                   # 95% CIs

    # --- output -------------------------------------------------------------
    "predictions_csv": "predictions_best.csv",   # best-on-val variant
    "summary_csv": "summary.csv",

    # --- synthetic generator (only used in synthetic mode) ------------------
    "synth_n_patients": 600,
    "synth_dim": 32,
    "synth_mean_lesions": 15,
    "synth_malignant_patient_frac": 0.18,
    "dump_synthetic_csv": "",           # optional: write the synthetic frame here
}

CONTEXT_METHODS = ("centroid", "cosine_centroid", "knn")

# common string -> 0/1 label mappings (lower-cased, stripped)
LABEL_MAP = {
    "0": 0, "1": 1,
    "benign": 0, "malignant": 1,
    "false": 0, "true": 1,
    "no": 0, "yes": 1,
}


# ============================================================================ #
# Small numeric helpers                                                        #
# ============================================================================ #
class MinMaxNorm:
    """Min-max normalizer FIT ON TRAIN ONLY, applied to val/test (no transduction)."""
    def __init__(self):
        self.lo = 0.0
        self.hi = 1.0
        self.const = False

    def fit(self, x: np.ndarray) -> "MinMaxNorm":
        x = np.asarray(x, dtype=float)
        self.lo = float(np.nanmin(x))
        self.hi = float(np.nanmax(x))
        self.const = (self.hi - self.lo) < 1e-12
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.const:                      # degenerate train range -> neutral constant
            return np.full(x.shape[0], 0.5, dtype=float)
        return np.clip((x - self.lo) / (self.hi - self.lo), 0.0, 1.0)


def _within_patient_ranks(patient_ids: np.ndarray, scores: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Rank lesions 1..m inside each patient by score (1 = highest = review first).

    Ties are broken with tiny deterministic jitter so no method gets a free
    advantage from equal scores.
    """
    scores = np.asarray(scores, dtype=float)
    jitter = rng.uniform(0.0, 1e-9, size=scores.shape[0])
    s = scores + jitter
    ranks = np.empty(scores.shape[0], dtype=int)
    order = np.argsort(patient_ids, kind="mergesort")
    pid_sorted = patient_ids[order]
    boundaries = np.flatnonzero(np.r_[True, pid_sorted[1:] != pid_sorted[:-1]])
    groups = np.split(order, boundaries[1:])
    for idx in groups:
        sub = s[idx]
        local = np.argsort(-sub, kind="mergesort")  # high score first
        r = np.empty(len(idx), dtype=int)
        r[local] = np.arange(1, len(idx) + 1)
        ranks[idx] = r
    return ranks


# ============================================================================ #
# Labels: strict validation + mapping                                          #
# ============================================================================ #
def validate_and_map_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Map labels to 0/1 with no silent coercion; stop with a clear error otherwise."""
    s = df[label_col]
    if s.isna().any():
        sys.exit(f"[FATAL] label column '{label_col}' contains missing/blank values; "
                 "cannot map to 0/1.")

    if pd.api.types.is_bool_dtype(s):
        mapped = s.astype(int).to_numpy()
    elif pd.api.types.is_numeric_dtype(s):
        vals = s.to_numpy()
        if not np.isin(vals, [0, 1]).all():
            bad = sorted(set(np.unique(vals).tolist()) - {0, 1, 0.0, 1.0})
            sys.exit(f"[FATAL] label column '{label_col}' has non-binary numeric "
                     f"values {bad}; expected only 0/1.")
        mapped = vals.astype(int)
    else:
        norm = s.astype(str).str.strip().str.lower()
        unmapped = sorted(set(norm.unique()) - set(LABEL_MAP))
        if unmapped:
            sys.exit(f"[FATAL] label column '{label_col}' has values that cannot be "
                     f"mapped to 0/1: {unmapped}. Supported (case-insensitive): "
                     f"{sorted(LABEL_MAP)}.")
        mapped = norm.map(LABEL_MAP).to_numpy().astype(int)

    df = df.copy()
    df[label_col] = mapped
    n0 = int((mapped == 0).sum())
    n1 = int((mapped == 1).sum())
    print(f"[labels] final counts after mapping -> benign/0: {n0}, malignant/1: {n1}")
    if n1 == 0:
        print("  !  NOTE: no malignant lesions present; top-k metrics over malignant "
              "patients will be empty (NaN).")
    return df


# ============================================================================ #
# Data: synthetic generator + real loader + checks                             #
# ============================================================================ #
def make_synthetic(cfg: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic cohort where malignant lesions are real per-patient outliers."""
    n_patients = cfg["synth_n_patients"]
    dim = cfg["synth_dim"]
    emb_cols = [f"{cfg['embedding_prefix']}{i}" for i in range(dim)]

    global_mal_dir = rng.normal(size=dim)
    global_mal_dir /= np.linalg.norm(global_mal_dir)

    rows = []
    lesion_counter = 0
    n_mal_patients = int(round(n_patients * cfg["synth_malignant_patient_frac"]))
    malignant_patients = set(rng.choice(n_patients, size=n_mal_patients, replace=False).tolist())

    for p in range(n_patients):
        n_lesions = int(rng.poisson(cfg["synth_mean_lesions"])) + 3   # >= 3
        patient_center = rng.normal(scale=1.5, size=dim)
        Z = patient_center + rng.normal(scale=0.5, size=(n_lesions, dim))
        labels = np.zeros(n_lesions, dtype=int)

        if p in malignant_patients:
            mal_idx = int(rng.integers(0, n_lesions))
            out_dir = rng.normal(size=dim)
            out_dir /= np.linalg.norm(out_dir)
            Z[mal_idx] = (patient_center
                          + 3.2 * out_dir                 # ugly-duckling deviation (context signal)
                          + 2.8 * global_mal_dir          # global signal the classifier can learn
                          + rng.normal(scale=0.35, size=dim))
            labels[mal_idx] = 1

        for j in range(n_lesions):
            row = {cfg["patient_id_col"]: f"P{p:04d}",
                   cfg["lesion_id_col"]: f"L{lesion_counter:06d}",
                   cfg["label_col"]: int(labels[j])}
            row.update({emb_cols[d]: float(Z[j, d]) for d in range(dim)})
            rows.append(row)
            lesion_counter += 1

    return pd.DataFrame(rows), emb_cols


def detect_embedding_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if str(c).startswith(prefix)]


def load_real(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    path = cfg["data_csv"]
    if not os.path.exists(path):
        sys.exit(f"[FATAL] --data_csv not found: {path}")
    df = pd.read_csv(path)
    return df, detect_embedding_cols(df, cfg["embedding_prefix"])


def check_required_columns(df: pd.DataFrame, cfg: dict, emb_cols: list[str]) -> None:
    pid, lid, lab = cfg["patient_id_col"], cfg["lesion_id_col"], cfg["label_col"]

    def mark(ok: bool) -> str:
        return "OK  " if ok else "MISSING"

    has_pid, has_lid, has_lab = pid in df.columns, lid in df.columns, lab in df.columns
    print(f"  [{mark(has_pid)}] patient_id_col  = '{pid}'")
    print(f"  [{mark(has_lid)}] lesion_id_col   = '{lid}'")
    print(f"  [{mark(has_lab)}] label_col       = '{lab}'")
    print(f"  embedding columns found (prefix '{cfg['embedding_prefix']}'): {len(emb_cols)}")
    fatal = False
    if not (has_pid and has_lid and has_lab):
        print("  -> required column(s) missing. Fix --patient_id_col / "
              "--lesion_id_col / --label_col.")
        fatal = True
    if len(emb_cols) == 0:
        print("  -> no embedding columns matched. Fix --embedding_prefix.")
        fatal = True
    if fatal:
        sys.exit("[FATAL] real-data checklist failed — see above.")


def real_data_hygiene(df: pd.DataFrame, cfg: dict, emb_cols: list[str]) -> pd.DataFrame:
    """NaN-embedding stop/drop, duplicate-id warn, <2-lesion-patient warn."""
    pid, lid = cfg["patient_id_col"], cfg["lesion_id_col"]
    Z = df[emb_cols].to_numpy(dtype=float)
    nan_rows = np.isnan(Z).any(axis=1)
    n_nan = int(nan_rows.sum())
    if n_nan > 0:
        if cfg["allow_drop_nan"]:
            print(f"  !  {n_nan} lesion row(s) have NaN embeddings -> DROPPED (--allow_drop_nan true).")
            df = df.loc[~nan_rows].reset_index(drop=True)
        else:
            sys.exit(f"[FATAL] {n_nan} lesion row(s) have NaN embedding value(s). "
                     "Clean the CSV, or pass --allow_drop_nan true to drop them.")

    n_dup = int(df[lid].duplicated().sum())
    if n_dup > 0:
        print(f"  !  WARNING: {n_dup} duplicate '{lid}' value(s) — lesion ids are not unique.")

    counts = df.groupby(pid)[lid].size()
    n_low = int((counts < 2).sum())
    if n_low > 0:
        print(f"  !  WARNING: {n_low} patient(s) have <2 lesions; context scoring is not "
              "meaningful for them (they receive context score 0.0).")
    return df


def cohort_report(df: pd.DataFrame, cfg: dict) -> None:
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    labels = df[lab].to_numpy().astype(int)
    n_patients = df[pid].nunique()
    n_lesions = len(df)
    n_mal_lesions = int((labels == 1).sum())
    mal_patients = df.loc[labels == 1, pid].nunique()
    print(f"  unique patients          : {n_patients}")
    print(f"  lesions (rows)           : {n_lesions}")
    print(f"  malignant lesions        : {n_mal_lesions}")
    print(f"  malignant patients       : {mal_patients}")
    if mal_patients < 5:
        print("  !! WARNING: VERY FEW malignant patients (<5). Top-k metrics and bootstrap "
              "CIs will be extremely unstable; treat results as a smoke test only.")
    elif mal_patients < 10:
        print("  !  CAUTION: few malignant patients (<10). Per-split positive counts will "
              "be small; widen splits or pool folds for stable CIs.")


# ============================================================================ #
# Splitting (patient-disjoint, stratified, with a hard leakage assertion)      #
# ============================================================================ #
def patient_disjoint_split(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    rng = np.random.default_rng(cfg["split_seed"])

    labels = df[lab].to_numpy().astype(int)
    per_patient_pos = df.assign(_y=labels).groupby(pid)["_y"].max()
    pos_patients = per_patient_pos.index[per_patient_pos.to_numpy() == 1].to_numpy()
    neg_patients = per_patient_pos.index[per_patient_pos.to_numpy() == 0].to_numpy()
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def carve(arr: np.ndarray):
        n = len(arr)
        n_test = int(round(n * cfg["test_frac"]))
        n_val = int(round(n * cfg["val_frac"]))
        return arr[n_test + n_val:], arr[n_test:n_test + n_val], arr[:n_test]

    tr_p, va_p, te_p = carve(pos_patients)
    tr_n, va_n, te_n = carve(neg_patients)
    train_p = set(np.concatenate([tr_p, tr_n]).tolist())
    val_p = set(np.concatenate([va_p, va_n]).tolist())
    test_p = set(np.concatenate([te_p, te_n]).tolist())

    assert train_p.isdisjoint(val_p), "LEAKAGE: train/val share patients"
    assert train_p.isdisjoint(test_p), "LEAKAGE: train/test share patients"
    assert val_p.isdisjoint(test_p), "LEAKAGE: val/test share patients"

    pid_arr = df[pid].to_numpy()
    return {
        "train": np.flatnonzero(np.isin(pid_arr, list(train_p))),
        "val": np.flatnonzero(np.isin(pid_arr, list(val_p))),
        "test": np.flatnonzero(np.isin(pid_arr, list(test_p))),
    }


# ============================================================================ #
# Models: classifier (with single-class guard) + context scores               #
# ============================================================================ #
class ConstantClassifier:
    """Fallback when the training split has a single class. Constant probability."""
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def fit_classifier(cfg: dict, Ztr: np.ndarray, ytr: np.ndarray):
    """Fit classifier; if train is single-class, return a constant scorer (no crash)."""
    if len(np.unique(ytr)) < 2:
        only = int(np.unique(ytr)[0]) if len(ytr) else -1
        print(f"  !! WARNING: training split has a SINGLE class (only label {only} present). "
              "Logistic regression disabled; returning CONSTANT classifier scores. "
              "context-only and random baselines still run.")
        base = float(np.mean(ytr)) if len(ytr) else 0.5
        return ConstantClassifier(p=base), False
    if cfg["classifier"] == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=tuple(cfg["mlp_hidden"]),
                            max_iter=300, random_state=cfg["split_seed"])
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Ztr, ytr)
    return clf, True


def context_scores(Z: np.ndarray, patient_ids: np.ndarray, method: str,
                   k: int, min_lesions: int) -> np.ndarray:
    """Per-patient unsupervised outlier score (higher = more 'ugly duckling')."""
    n = Z.shape[0]
    scores = np.zeros(n, dtype=float)
    order = np.argsort(patient_ids, kind="mergesort")
    pid_sorted = patient_ids[order]
    boundaries = np.flatnonzero(np.r_[True, pid_sorted[1:] != pid_sorted[:-1]])
    groups = np.split(order, boundaries[1:])

    for idx in groups:
        m = len(idx)
        if m < min_lesions:
            continue
        Zi = Z[idx]
        if method == "centroid":
            c = Zi.mean(axis=0, keepdims=True)
            d = np.linalg.norm(Zi - c, axis=1)
        elif method == "cosine_centroid":
            c = Zi.mean(axis=0, keepdims=True)
            num = (Zi * c).sum(axis=1)
            den = (np.linalg.norm(Zi, axis=1) * np.linalg.norm(c) + 1e-12)
            d = 1.0 - num / den
        elif method == "knn":
            keff = min(k, m - 1)
            diff = Zi[:, None, :] - Zi[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            d = np.sort(dist, axis=1)[:, :keff].mean(axis=1)
        else:
            raise ValueError(f"unknown context method: {method}")
        scores[idx] = d
    return scores


# ============================================================================ #
# Metrics                                                                      #
# ============================================================================ #
def topk_metrics(patient_ids, labels, scores, k_list, rng) -> dict:
    """Patient-level top-k triage metrics, computed over MALIGNANT patients."""
    ranks = _within_patient_ranks(patient_ids, scores, rng)
    recalls = {k: [] for k in k_list}
    rank_first, pct_rank = [], []

    order = np.argsort(patient_ids, kind="mergesort")
    pid_sorted = patient_ids[order]
    boundaries = np.flatnonzero(np.r_[True, pid_sorted[1:] != pid_sorted[:-1]])
    groups = np.split(order, boundaries[1:])

    for idx in groups:
        y = labels[idx]
        if y.sum() == 0:
            continue
        m = len(idx)
        rfm = int(ranks[idx][y == 1].min())
        rank_first.append(rfm)
        pct_rank.append(rfm / m)
        for k in k_list:
            recalls[k].append(1.0 if rfm <= k else 0.0)

    n_mal = len(rank_first)
    out = {f"recall@{k}": (float(np.mean(recalls[k])) if n_mal else float("nan")) for k in k_list}
    out["mean_rank_first_malignant"] = float(np.mean(rank_first)) if n_mal else float("nan")
    out["median_rank_first_malignant"] = float(np.median(rank_first)) if n_mal else float("nan")
    out["mean_percentile_rank_first_malignant"] = float(np.mean(pct_rank)) if n_mal else float("nan")
    out["mean_NNR_to_first_malignant"] = out["mean_rank_first_malignant"]  # NNTR = mean RFM
    out["n_malignant_patients_evaluated"] = n_mal
    return out


def secondary_lesion_metrics(labels, scores) -> dict:
    """Lesion-level AUROC / AUPRC — SECONDARY references only."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def bootstrap_ci(patient_ids, labels, scores, k_list, n_boot, alpha, seed) -> dict:
    """Patient-level bootstrap (resample PATIENTS, not lesions)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(patient_ids)
    idx_by_pid = {p: np.flatnonzero(patient_ids == p) for p in uniq}
    keys = [f"recall@{k}" for k in k_list] + ["mean_rank_first_malignant"]
    collected = {kk: [] for kk in keys}

    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_pid[p] for p in sample])
        remap = np.repeat(np.arange(len(sample)), [len(idx_by_pid[p]) for p in sample])
        m = topk_metrics(remap, labels[sel], scores[sel], k_list, np.random.default_rng(0))
        for kk in keys:
            collected[kk].append(m[kk])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {kk: (float(np.nanpercentile(v, lo_q)), float(np.nanpercentile(v, hi_q)))
            for kk, v in collected.items()}


def random_baseline_metrics(patient_ids, labels, k_list, n_seeds, seed) -> dict:
    """Within-patient RANDOM ranking, averaged over seeds (chance reference)."""
    rng = np.random.default_rng(seed)
    runs = [topk_metrics(patient_ids, labels, rng.uniform(size=labels.shape[0]),
                         k_list, np.random.default_rng(0)) for _ in range(n_seeds)]
    numeric = [f"recall@{k}" for k in k_list] + [
        "mean_rank_first_malignant", "median_rank_first_malignant",
        "mean_percentile_rank_first_malignant", "mean_NNR_to_first_malignant"]
    out = {key: float(np.mean([r[key] for r in runs])) for key in numeric}
    out["n_malignant_patients_evaluated"] = runs[0]["n_malignant_patients_evaluated"]
    return out


def review_burden(patient_ids, labels, scores, k_list, rng) -> dict:
    """Review-burden / benign-patient prioritisation (all patients)."""
    ranks = _within_patient_ranks(patient_ids, scores, rng)
    uniq = np.unique(patient_ids)
    sizes = np.array([(patient_ids == p).sum() for p in uniq], dtype=float)
    budget_pct = {k: float(np.mean(np.minimum(k, sizes) / sizes)) for k in k_list}
    top_scores = []
    for p in uniq:
        idx = np.flatnonzero(patient_ids == p)
        if labels[idx].sum() == 0:
            top_scores.append(float(scores[idx][ranks[idx] == 1][0]))
    return {"mean_lesions_per_patient": float(sizes.mean()),
            "topk_budget_pct": budget_pct,
            "benign_patient_mean_top_score": float(np.mean(top_scores)) if top_scores else float("nan")}


# ============================================================================ #
# Fusion weight selection                                                      #
# ============================================================================ #
def select_weight(val_pid, val_y, clf_norm_val, ctx_norm_val, cfg) -> float:
    """Pick classifier weight w on validation by recall@3 (tie-break lower MRFM)."""
    best_w, best_key = None, None
    rng = np.random.default_rng(cfg["split_seed"])
    sel = cfg["select_metric"]
    for w in cfg["weight_grid"]:
        combined = w * clf_norm_val + (1.0 - w) * ctx_norm_val
        m = topk_metrics(val_pid, val_y, combined, cfg["k_list"], rng)
        key = (m.get(sel, float("nan")), -m["mean_rank_first_malignant"])
        if best_key is None or key > best_key:
            best_key, best_w = key, w
    return float(best_w)


# ============================================================================ #
# Prediction-frame assembly (raw + normalized scores)                          #
# ============================================================================ #
def build_predictions(df_split, cfg, clf_raw, clf_norm, ctx_raw, ctx_norm,
                      combined, weight, method_name, split_name, rng) -> pd.DataFrame:
    pid_arr = df_split[cfg["patient_id_col"]].to_numpy()
    ranks = _within_patient_ranks(pid_arr, combined, rng)
    out = pd.DataFrame({
        "patient_id": df_split[cfg["patient_id_col"]].to_numpy(),
        "lesion_id": df_split[cfg["lesion_id_col"]].to_numpy(),
        "true_label": df_split[cfg["label_col"]].to_numpy().astype(int),
        "classifier_score_raw": clf_raw,
        "classifier_score_norm": clf_norm,
        "context_score_raw": ctx_raw,
        "context_score_norm": ctx_norm,
        "combined_score": combined,
        "patient_rank": ranks,
        "context_method": method_name,
        "selected_weight": weight,
        "split": split_name,
    })
    return out.sort_values(["patient_id", "patient_rank"]).reset_index(drop=True)


# ============================================================================ #
# Main                                                                         #
# ============================================================================ #
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Patient-level lesion-triage baseline (V2.2). "
                    "Top-k / NNTR is the primary metric, not AUROC.")

    def b(s: str) -> bool:
        return str(s).strip().lower() in ("1", "true", "t", "yes", "y")

    p.add_argument("--data_csv", type=str)
    p.add_argument("--use_synthetic_data", type=b)
    p.add_argument("--patient_id_col", type=str)
    p.add_argument("--lesion_id_col", type=str)
    p.add_argument("--label_col", type=str)
    p.add_argument("--embedding_prefix", type=str)
    p.add_argument("--allow_drop_nan", type=b,
                   help="real data: drop NaN-embedding rows instead of stopping (default false)")
    p.add_argument("--context_method", type=str,
                   choices=["all", "centroid", "cosine_centroid", "knn"])
    p.add_argument("--predictions_csv", type=str,
                   help="path for predictions_best.csv (per-method files written alongside)")
    p.add_argument("--summary_csv", type=str)
    p.add_argument("--split_seed", type=int)
    p.add_argument("--n_bootstrap", type=int)
    p.add_argument("--n_random_seeds", type=int)
    p.add_argument("--context_k", type=int)
    p.add_argument("--dump_synthetic_csv", type=str,
                   help="(testing) write the synthetic cohort to this CSV path")

    args = vars(p.parse_args())
    cfg = dict(CONFIG)
    for k, v in args.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg["split_seed"])

    print("=" * 78)
    print("lesion_baseline.py  v2.2   |   primary metric: top-k / NNTR (not AUROC)")
    print("=" * 78)

    # ---- 1. data + validation ---------------------------------------------
    if cfg["use_synthetic_data"]:
        df, emb_cols = make_synthetic(cfg, rng)
        print(f"[data] SYNTHETIC cohort: {df[cfg['patient_id_col']].nunique()} patients, "
              f"{len(df)} lesions, {len(emb_cols)} embedding dims.")
        df = validate_and_map_labels(df, cfg["label_col"])
        if cfg["dump_synthetic_csv"]:
            df.to_csv(cfg["dump_synthetic_csv"], index=False)
            print(f"[data] wrote synthetic CSV -> {cfg['dump_synthetic_csv']}")
    else:
        df, emb_cols = load_real(cfg)
        print("\n" + "=" * 78)
        print("REAL-DATA CHECKLIST  (--use_synthetic_data false)")
        print("=" * 78)
        print(f"  file: {cfg['data_csv']}")
        check_required_columns(df, cfg, emb_cols)      # fatal if missing
        df = validate_and_map_labels(df, cfg["label_col"])  # strict labels + counts
        df = real_data_hygiene(df, cfg, emb_cols)      # NaN/dup/low-lesion
        cohort_report(df, cfg)                         # counts + few-positive warnings
        print("=" * 78 + "\n")

    df = df.reset_index(drop=True)
    Z_all = df[emb_cols].to_numpy(dtype=float)
    y_all = df[cfg["label_col"]].to_numpy().astype(int)

    # ---- 2. patient-disjoint split ----------------------------------------
    splits = patient_disjoint_split(df, cfg)
    for name in ("train", "val", "test"):
        sub = df.iloc[splits[name]]
        print(f"[split] {name:5s}: {sub[cfg['patient_id_col']].nunique():4d} patients, "
              f"{len(sub):5d} lesions, {int(y_all[splits[name]].sum()):3d} malignant lesions")

    # ---- 3. standardise on TRAIN only -------------------------------------
    scaler = StandardScaler().fit(Z_all[splits["train"]])
    Z = scaler.transform(Z_all)

    SPL = ("train", "val", "test")
    pid = {s: df.iloc[splits[s]][cfg["patient_id_col"]].to_numpy() for s in SPL}
    yv = {s: y_all[splits[s]] for s in SPL}

    # ---- 4. classifier (single-class guarded) + TRAIN-fit normalizer ------
    clf, clf_is_real = fit_classifier(cfg, Z[splits["train"]], y_all[splits["train"]])
    clf_raw = {s: clf.predict_proba(Z[splits[s]])[:, 1] for s in SPL}
    clf_norm_model = MinMaxNorm().fit(clf_raw["train"])           # FIT ON TRAIN ONLY
    clf_norm = {s: clf_norm_model.transform(clf_raw[s]) for s in SPL}

    methods = list(CONTEXT_METHODS) if cfg["context_method"] == "all" else [cfg["context_method"]]

    # ---- 5. context scores for train/val/test; normalizer fit on TRAIN ----
    ctx_raw, ctx_norm = {}, {}
    for m in methods:
        ctx_raw[m] = {s: context_scores(Z[splits[s]], pid[s], m,
                                         cfg["context_k"], cfg["min_lesions_for_context"])
                      for s in SPL}
        ctx_norm_model = MinMaxNorm().fit(ctx_raw[m]["train"])    # FIT ON TRAIN ONLY
        ctx_norm[m] = {s: ctx_norm_model.transform(ctx_raw[m][s]) for s in SPL}

    # ---- 6. fusion weight selection on validation -------------------------
    weights = {m: select_weight(pid["val"], yv["val"], clf_norm["val"], ctx_norm[m]["val"], cfg)
               for m in methods}

    def combined_score(m, s):
        w = weights[m]
        return w * clf_norm[s] + (1.0 - w) * ctx_norm[m][s]

    # best variant on VALIDATION recall@3 (tie-break lower MRFM)
    val_variants = {"classifier_only": clf_norm["val"]}
    for m in methods:
        val_variants[f"combined[{m}]"] = combined_score(m, "val")
    best_label, best_key = None, None
    for label, sc in val_variants.items():
        mm = topk_metrics(pid["val"], yv["val"], sc, cfg["k_list"], rng)
        key = (mm["recall@3"], -mm["mean_rank_first_malignant"])
        if best_key is None or key > best_key:
            best_key, best_label = key, label

    # ---- 7. summary rows (classifier_only, random, per-method) ------------
    rows, secondary = [], {}

    def add_row(label, split_name, metrics):
        row = {"method": label, "split": split_name}
        for col in ("recall@1", "recall@3", "recall@5", "mean_rank_first_malignant",
                    "median_rank_first_malignant", "mean_percentile_rank_first_malignant",
                    "mean_NNR_to_first_malignant", "n_malignant_patients_evaluated"):
            row[col] = metrics.get(col, float("nan"))
        rows.append(row)

    for split_name in ("val", "test"):
        P, Y = pid[split_name], yv[split_name]
        add_row("classifier_only", split_name,
                topk_metrics(P, Y, clf_norm[split_name], cfg["k_list"], rng))
        secondary[("classifier_only", split_name)] = secondary_lesion_metrics(Y, clf_raw[split_name])
        add_row("random", split_name,
                random_baseline_metrics(P, Y, cfg["k_list"], cfg["n_random_seeds"], cfg["split_seed"]))
        for m in methods:
            add_row(f"context_only[{m}]", split_name,
                    topk_metrics(P, Y, ctx_norm[m][split_name], cfg["k_list"], rng))
            secondary[(f"context_only[{m}]", split_name)] = \
                secondary_lesion_metrics(Y, ctx_raw[m][split_name])
            comb = combined_score(m, split_name)
            add_row(f"combined[{m}]", split_name, topk_metrics(P, Y, comb, cfg["k_list"], rng))
            secondary[(f"combined[{m}]", split_name)] = secondary_lesion_metrics(Y, comb)

    summary = pd.DataFrame(rows, columns=[
        "method", "split", "recall@1", "recall@3", "recall@5",
        "mean_rank_first_malignant", "median_rank_first_malignant",
        "mean_percentile_rank_first_malignant", "mean_NNR_to_first_malignant",
        "n_malignant_patients_evaluated"])
    summary.to_csv(cfg["summary_csv"], index=False)

    # ---- 8. console report -------------------------------------------------
    print("\n" + "-" * 78)
    print("SUMMARY  (rows = method x split; primary metric block first)")
    print("-" * 78)
    with pd.option_context("display.width", 170, "display.max_columns", 20):
        show = summary.copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
        print(show.to_string(index=False))

    print("\nSECONDARY lesion-level references (NOT the primary metric):")
    for split_name in ("val", "test"):
        for (variant, sp), sm in secondary.items():
            if sp == split_name:
                print(f"  [{split_name}] {variant:24s}  AUROC={sm['auroc']:.3f}  AUPRC={sm['auprc']:.3f}")

    print("\nPATIENT-LEVEL BOOTSTRAP 95% CIs on TEST "
          f"({cfg['n_bootstrap']} resamples of patients):")
    P, Y = pid["test"], yv["test"]
    ci_variants = [("classifier_only", clf_norm["test"])] + \
                  [(f"combined[{m}]", combined_score(m, "test")) for m in methods]
    for label, sc in ci_variants:
        ci = bootstrap_ci(P, Y, sc, cfg["k_list"], cfg["n_bootstrap"], cfg["ci_alpha"], cfg["split_seed"])
        bits = ", ".join(f"{k}=[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in ci.items())
        print(f"  {label:22s} {bits}")

    # ---- review burden: classifier_only, best combined, each method -------
    print("\nREVIEW BURDEN / benign-patient prioritisation (TEST):")
    rb0 = review_burden(P, Y, clf_norm["test"], cfg["k_list"], rng)
    print(f"  mean lesions/patient: {rb0['mean_lesions_per_patient']:.2f}")
    print("  top-k budget as % of each patient's lesions reviewed: "
          + ", ".join(f"k={k}:{v*100:.1f}%" for k, v in rb0["topk_budget_pct"].items()))
    print("  benign-only patients, mean top-ranked lesion score (lower = better triage):")
    print(f"    classifier_only        : {rb0['benign_patient_mean_top_score']:.3f}")
    for m in methods:
        rb = review_burden(P, Y, combined_score(m, "test"), cfg["k_list"], rng)
        tag = "  <- best-selected" if best_label == f"combined[{m}]" else ""
        print(f"    combined[{m:16s}]: {rb['benign_patient_mean_top_score']:.3f}{tag}")

    # ---- 9. predictions files (TEST split, raw + normalized) --------------
    out_dir = os.path.dirname(os.path.abspath(cfg["predictions_csv"]))
    os.makedirs(out_dir, exist_ok=True)
    written = []
    per_method_frames = {}
    for m in methods:
        frame = build_predictions(
            df.iloc[splits["test"]], cfg,
            clf_raw["test"], clf_norm["test"], ctx_raw[m]["test"], ctx_norm[m]["test"],
            combined_score(m, "test"), weights[m], m, "test",
            np.random.default_rng(cfg["split_seed"]))
        per_method_frames[m] = frame
        path = os.path.join(out_dir, f"predictions_{m}.csv")
        frame.to_csv(path, index=False)
        written.append(path)

    if best_label == "classifier_only":
        nan_col = np.full(len(splits["test"]), np.nan)
        best_frame = build_predictions(
            df.iloc[splits["test"]], cfg,
            clf_raw["test"], clf_norm["test"], nan_col, nan_col,
            clf_norm["test"], 1.0, "classifier_only", "test",
            np.random.default_rng(cfg["split_seed"]))
    else:
        best_frame = per_method_frames[best_label.split("[")[1].rstrip("]")].copy()
    best_frame.to_csv(cfg["predictions_csv"], index=False)
    written.append(cfg["predictions_csv"])

    print(f"\n[select] best variant on validation recall@3: {best_label} "
          f"-> {os.path.basename(cfg['predictions_csv'])}")
    print("\nFILES WRITTEN:")
    print(f"  {cfg['summary_csv']}")
    for w_ in written:
        print(f"  {w_}")

    # ---- copy-paste real-data command -------------------------------------
    print("\n" + "=" * 78)
    print("RUN ON REAL DATA (copy/paste, then edit YOUR_FILE.csv + column names):")
    print("=" * 78)
    print("python lesion_baseline.py --use_synthetic_data false "
          "--data_csv YOUR_FILE.csv --patient_id_col patient_id "
          "--lesion_id_col lesion_id --label_col malignant "
          "--embedding_prefix emb_ --context_method all")
    print("=" * 78)


if __name__ == "__main__":
    main()
