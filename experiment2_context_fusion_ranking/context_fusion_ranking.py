#!/usr/bin/env python3
"""
context_fusion_ranking.py  -  Experiment 2
==========================================
Stronger, statistically-defensible patient-context lesion ranking.

Research question
-----------------
Under a limited clinician review budget, does fusing patient-context features
with the supervised lesion risk score surface each patient's malignant lesion
EARLIER than classifier-only ranking?

This is the SECOND-STAGE ranking experiment only. There is no high-resolution
detection, no YOLO, no transformer, no segmentation here. Lesion-level AUROC /
AUPRC are reported but are SECONDARY; the primary metrics are patient-level
top-k review-budget metrics.

Pipeline (all leakage-controlled, patient-disjoint):
  1. patient-disjoint train / val / test split
  2. first-stage classifier on embeddings (fit on TRAIN only)
  3. per-patient context features (transductive: a test patient may use its own
     lesion set, but never test labels)
  4. six ranking methods: random, classifier_only, context_only, manual_fusion,
     learned_fusion_logreg, (optional) learned_fusion_mlp
  5. manual-fusion weight and the "best" method are selected on VALIDATION;
     final numbers are reported on held-out TEST
  6. paired patient-level bootstrap of every method vs classifier_only
  7. write summary / predictions / paired report / readable report

CLI example
-----------
python context_fusion_ranking.py \
    --data_csv ../experiment1_contextual_lesion_prio/data/lesions_embeddings.csv \
    --patient_id_col patient_id --lesion_id_col lesion_id --label_col malignant \
    --embedding_prefix emb_ --split_seed 42 --output_dir results/seed_42 \
    --n_bootstrap 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import paired_patient_bootstrap as ppb


# ----------------------------------------------------------------------------- #
# CONFIG (defaults; the YAML and then the CLI override these in that order)
# ----------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "data_csv": "data/lesions_embeddings.csv",
    "patient_id_col": "patient_id",
    "lesion_id_col": "lesion_id",
    "label_col": "malignant",
    "embedding_prefix": "emb_",
    "use_synthetic_data": False,

    "val_frac": 0.20,
    "test_frac": 0.20,
    "split_seed": 42,

    "classifier": "logreg",
    "mlp_hidden": [64],

    "context_k": 5,
    "min_lesions_for_context": 2,
    "use_isolation_forest": True,
    "use_lof": True,

    "methods": ["random", "classifier_only", "context_only",
                "manual_fusion", "learned_fusion_logreg", "learned_fusion_mlp"],
    "weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],

    "topk_values": [1, 3, 5, 10],
    "selection_metric": "recall@5",
    "n_bootstrap": 1000,

    "output_dir": "results/seed_42",

    "syn_n_patients": 509,
    "syn_embed_dim": 64,
    "syn_malignant_patient_frac": 0.5,
    "syn_mean_lesions": 23,
}


# ----------------------------------------------------------------------------- #
# Config loading + CLI overrides
# ----------------------------------------------------------------------------- #
def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    return cfg


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_args_into_config():
    p = argparse.ArgumentParser(description="Experiment 2: context fusion ranking.")
    p.add_argument("--config", type=str,
                   default="configs/slice3d_context_fusion.yaml")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--patient_id_col", type=str, default=None)
    p.add_argument("--lesion_id_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--embedding_prefix", type=str, default=None)
    p.add_argument("--use_synthetic_data", type=_str2bool, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--n_bootstrap", type=int, default=None)
    p.add_argument("--context_k", type=int, default=None)
    p.add_argument("--selection_metric", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    for key in ["data_csv", "patient_id_col", "lesion_id_col", "label_col",
                "embedding_prefix", "use_synthetic_data", "split_seed",
                "output_dir", "n_bootstrap", "context_k", "selection_metric"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    return cfg


# ----------------------------------------------------------------------------- #
# Synthetic data (smoke-test only; mirrors the SLICE-3D CSV shape)
# ----------------------------------------------------------------------------- #
def generate_synthetic_data(cfg):
    """Toy SLICE-3D-shaped data with COMPLEMENTARY signal, so fusion can beat
    either single method (mirrors the real result):
      * malignant lesions are shifted along a global, classifier-learnable
        direction (mal_dir) AND are per-patient outliers;
      * some benign lesions ('nevus outliers') are per-patient outliers too,
        which are context false-positives the classifier can ignore.
    """
    rng = np.random.default_rng(cfg["split_seed"])
    dim = cfg["syn_embed_dim"]
    mal_dir = rng.normal(0, 1, dim); mal_dir /= np.linalg.norm(mal_dir)
    rows, lc = [], 0
    for p in range(cfg["syn_n_patients"]):
        pid = f"P{p:04d}"
        n = max(1, rng.poisson(cfg["syn_mean_lesions"]))
        mal_patient = rng.random() < cfg["syn_malignant_patient_frac"]
        n_mal = min(n, 1 + int(rng.random() < 0.3)) if mal_patient else 0
        center = rng.normal(0, 1.5, dim)
        for i in range(n):
            lab = 1 if i < n_mal else 0
            emb = center + rng.normal(0, 1.0, dim)
            if lab:
                rdir = rng.normal(0, 1, dim); rdir /= np.linalg.norm(rdir)
                emb = emb + 2.2 * mal_dir + 2.0 * rdir  # global signal + outlier
            elif rng.random() < 0.15:
                rdir = rng.normal(0, 1, dim); rdir /= np.linalg.norm(rdir)
                emb = emb + 2.5 * rdir                  # benign outlier (context FP)
            row = {cfg["patient_id_col"]: pid,
                   cfg["lesion_id_col"]: f"L{lc:06d}",
                   cfg["label_col"]: lab}
            row.update({f"{cfg['embedding_prefix']}{d}": emb[d] for d in range(dim)})
            rows.append(row); lc += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=cfg["split_seed"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- #
# Load + validate + real-data checklist
# ----------------------------------------------------------------------------- #
def resolve_embedding_cols(df, cfg):
    return [c for c in df.columns if c.startswith(cfg["embedding_prefix"])]


def load_data(cfg):
    if cfg["use_synthetic_data"]:
        print(">> use_synthetic_data=True -> generating SLICE-3D-shaped toy set.\n")
        df = generate_synthetic_data(cfg)
    elif not os.path.exists(cfg["data_csv"]):
        sys.exit(f"ERROR: data_csv '{cfg['data_csv']}' not found. Provide the "
                 "Experiment 1 prepared CSV, or set use_synthetic_data: true.")
    else:
        df = pd.read_csv(cfg["data_csv"])

    required = [cfg["patient_id_col"], cfg["lesion_id_col"], cfg["label_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing required column(s): {missing}")
    embed_cols = resolve_embedding_cols(df, cfg)
    if not embed_cols:
        sys.exit(f"ERROR: no embedding columns with prefix "
                 f"'{cfg['embedding_prefix']}' found.")

    n0 = len(df)
    df = df.dropna(subset=[cfg["label_col"]] + embed_cols).reset_index(drop=True)
    if len(df) < n0:
        print(f">> WARNING: dropped {n0 - len(df)} rows with NaN label/embedding.\n")
    df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)
    return df, embed_cols


def real_data_checklist(df, embed_cols, cfg):
    if cfg["use_synthetic_data"]:
        return
    pid, lid, lab = cfg["patient_id_col"], cfg["lesion_id_col"], cfg["label_col"]
    print("=" * 70)
    print("REAL-DATA CHECKLIST")
    print("=" * 70)
    chk = lambda ok, m: print(f"  [{'OK' if ok else 'XX'}] {m}")
    chk(pid in df.columns, f"patient_id_col '{pid}' present")
    chk(lid in df.columns, f"lesion_id_col '{lid}' present")
    chk(lab in df.columns, f"label_col '{lab}' present")
    chk(len(embed_cols) > 0, f"embedding columns found: {len(embed_cols)}")
    n_mal = int(df.groupby(pid)[lab].max().sum())
    print(f"  [..] unique patients           : {df[pid].nunique()}")
    print(f"  [..] malignant patients        : {n_mal}")
    if n_mal < 10:
        print(f"  [!!] WARNING: very few malignant patients ({n_mal}); "
              "patient-level metrics and bootstrap CIs will be noisy.")
    print()


def audit_dataset(df, embed_cols, cfg):
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    per = df.groupby(pid).size()
    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    print(f"embedding dim                 : {len(embed_cols)}")
    print(f"patients                      : {df[pid].nunique()}")
    print(f"lesions                       : {len(df)}")
    print(f"lesions/patient (min/med/max) : {per.min()} / {per.median():.0f} / {per.max()}")
    print(f"malignant lesions             : {int(df[lab].sum())} ({100*df[lab].mean():.2f}%)")
    print(f"patients w/ >=1 malignant     : {int(df.groupby(pid)[lab].max().sum())}")
    print()


# ----------------------------------------------------------------------------- #
# Patient-disjoint split (stratified on malignant-patient flag)
# ----------------------------------------------------------------------------- #
def patient_disjoint_split(df, cfg):
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    patients = df.groupby(pid)[lab].max().reset_index()
    ids, strat = patients[pid].values, patients[lab].values
    s = strat if (strat.sum() >= 2 and (strat == 0).sum() >= 2) else None

    test_size = cfg["test_frac"]
    val_size = cfg["val_frac"] / (1.0 - test_size)
    p_tv, p_te = train_test_split(ids, test_size=test_size,
                                  random_state=cfg["split_seed"], stratify=s)
    s_tv = (patients.set_index(pid).loc[p_tv, lab].values if s is not None else None)
    if s_tv is not None and (s_tv.sum() < 2 or (s_tv == 0).sum() < 2):
        s_tv = None
    p_tr, p_va = train_test_split(p_tv, test_size=val_size,
                                  random_state=cfg["split_seed"], stratify=s_tv)
    tr, va, te = set(p_tr), set(p_va), set(p_te)
    assert not (tr & va) and not (tr & te) and not (va & te), "PATIENT LEAKAGE"

    masks = {
        "train": df[pid].isin(tr).values,
        "val": df[pid].isin(va).values,
        "test": df[pid].isin(te).values,
    }
    print("=" * 70)
    print("PATIENT-DISJOINT SPLIT (leakage-checked)")
    print("=" * 70)
    for name, m in masks.items():
        d = df[m]
        print(f"{name:5s}: {d[pid].nunique():4d} patients | {len(d):6d} lesions | "
              f"{int(d.groupby(pid)[lab].max().sum()):4d} malignant patients")
    print()
    return masks


# ----------------------------------------------------------------------------- #
# Rank-normaliser (fit on TRAIN; monotonic map to [0,1])
# ----------------------------------------------------------------------------- #
class RankNormalizer:
    def __init__(self, ref):
        self.ref = np.sort(np.asarray(ref, float))

    def transform(self, x):
        x = np.asarray(x, float)
        return np.searchsorted(self.ref, x, side="right") / max(len(self.ref), 1)


# ----------------------------------------------------------------------------- #
# Context features (per-patient, transductive, label-free)
# ----------------------------------------------------------------------------- #
def compute_context_features(df, Xs, clf_score, train_mask, cfg):
    """Return a feature DataFrame aligned to df rows.

    Geometric + within-patient features use only a patient's own lesions (safe
    for test patients). IsolationForest / LOF are fit on TRAIN embeddings only
    (global novelty signal) so they never see val/test during fitting.
    """
    pid = cfg["patient_id_col"]
    n = len(df)
    feat = pd.DataFrame(index=df.index)
    feat["clf_score"] = clf_score

    # --- within-patient classifier-derived features ----------------------
    g = pd.Series(clf_score, index=df.index).groupby(df[pid])
    p_mean = g.transform("mean")
    p_std = g.transform("std").fillna(0.0)
    feat["clf_pct_rank_in_patient"] = g.rank(pct=True).values
    feat["clf_margin_from_patient_mean"] = clf_score - p_mean.values
    feat["clf_zscore_in_patient"] = np.where(
        p_std.values > 1e-9, (clf_score - p_mean.values) / (p_std.values + 1e-9), 0.0)
    feat["patient_mean_clf"] = p_mean.values
    feat["patient_std_clf"] = p_std.values

    # --- per-patient geometric / outlier features ------------------------
    centroid_d = np.zeros(n); cosine_d = np.zeros(n)
    knn_d = np.zeros(n); max_nn_d = np.zeros(n)
    kmin = cfg["min_lesions_for_context"]
    for _, grp in df.groupby(pid):
        rows = grp.index.to_numpy()
        X = Xs[df.index.get_indexer(rows)]
        if len(rows) < kmin:
            continue
        c = X.mean(axis=0)
        centroid_d[df.index.get_indexer(rows)] = np.linalg.norm(X - c, axis=1)
        d = c / (np.linalg.norm(c) + 1e-9)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cosine_d[df.index.get_indexer(rows)] = 1.0 - Xn @ d
        k = min(cfg["context_k"], len(rows) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, _ = nn.kneighbors(X)
        knn_d[df.index.get_indexer(rows)] = dist[:, 1:].mean(axis=1)
        max_nn_d[df.index.get_indexer(rows)] = dist[:, 1:].max(axis=1)
    feat["centroid_eucl_dist"] = centroid_d
    feat["cosine_centroid_dist"] = cosine_d
    feat["knn_dist"] = knn_d
    feat["max_nn_dist"] = max_nn_d

    # --- lesion count per patient ----------------------------------------
    feat["lesion_count"] = df.groupby(pid)[pid].transform("size").values

    # --- optional global outlier scores (fit on TRAIN embeddings only) ---
    Xtr = Xs[train_mask]
    if cfg["use_isolation_forest"]:
        iforest = IsolationForest(n_estimators=200, random_state=cfg["split_seed"]).fit(Xtr)
        feat["iforest_score"] = -iforest.score_samples(Xs)  # higher = more anomalous
    if cfg["use_lof"]:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr)
        feat["lof_score"] = -lof.score_samples(Xs)          # higher = more anomalous

    return feat


# ----------------------------------------------------------------------------- #
# Review-budget evaluation (PRIMARY) + lesion-level AUROC/AUPRC (SECONDARY)
# ----------------------------------------------------------------------------- #
def rank_within_patient(df, scores, cfg):
    out = df.copy()
    out["_score"] = scores
    out["patient_rank"] = (out.groupby(cfg["patient_id_col"])["_score"]
                           .rank(method="first", ascending=False).astype(int))
    return out


def review_budget_eval(ranked, cfg):
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    first, pctile = [], []
    hits = {k: [] for k in cfg["topk_values"]}
    for _, g in ranked.groupby(pid):
        if g[lab].sum() == 0:
            continue
        r = int(g.loc[g[lab] == 1, "patient_rank"].min())
        first.append(r); pctile.append(r / len(g))
        for k in cfg["topk_values"]:
            hits[k].append(1 if r <= k else 0)
    first = np.array(first); pctile = np.array(pctile)
    m = {f"recall@{k}": float(np.mean(hits[k])) if hits[k] else float("nan")
         for k in cfg["topk_values"]}
    m["mean_rank_first_malignant"] = float(first.mean()) if first.size else float("nan")
    m["median_rank_first_malignant"] = float(np.median(first)) if first.size else float("nan")
    m["mean_NNR_to_first_malignant"] = m["mean_rank_first_malignant"]
    m["mean_percentile_rank_first_malignant"] = float(pctile.mean()) if pctile.size else float("nan")
    m["median_percentile_rank_first_malignant"] = float(np.median(pctile)) if pctile.size else float("nan")
    m["n_malignant_patients_evaluated"] = int(first.size)
    return m


def review_burden(ranked, cfg):
    pid, lab = cfg["patient_id_col"], cfg["label_col"]
    sizes = ranked.groupby(pid).size()
    b = {"mean_lesions_per_patient": float(sizes.mean())}
    for k in cfg["topk_values"]:
        b[f"top{k}_pct_of_lesions_reviewed"] = float(
            100 * (np.minimum(k, sizes.values) / sizes.values).mean())
    benign_top = [float(g.loc[g["patient_rank"] == 1, "_score"].iloc[0])
                  for _, g in ranked.groupby(pid) if g[lab].sum() == 0]
    b["n_benign_only_patients"] = len(benign_top)
    b["benign_only_mean_top1_score"] = float(np.mean(benign_top)) if benign_top else float("nan")
    return b


def lesion_level_secondary(y, scores):
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores))}


# ----------------------------------------------------------------------------- #
# Method scores  (all return a per-lesion score; higher = review earlier)
# ----------------------------------------------------------------------------- #
def build_method_scores(df, feat, masks, cfg):
    """Produce a per-lesion score array (over ALL rows) for each method.

    Fitting rule: anything learned (manual-fusion weight, learned-fusion models,
    normalisers) is fit on TRAIN and selected on VAL; nothing peeks at TEST
    labels. Returns (scores_by_method, extras) where extras carries the chosen
    manual-fusion weight and the per-method val selection metric.
    """
    lab = cfg["label_col"]
    train_mask, val_mask = masks["train"], masks["val"]
    y_train = df[lab].values[train_mask]
    scores = {}
    extras = {"manual_fusion_weight": None}

    # 1. random ------------------------------------------------------------
    rng = np.random.default_rng(cfg["split_seed"])
    scores["random"] = rng.random(len(df))

    # 2. classifier_only ---------------------------------------------------
    scores["classifier_only"] = feat["clf_score"].values

    # 3. context_only (primary unsupervised signal = centroid distance) ----
    scores["context_only"] = feat["centroid_eucl_dist"].values

    # 4. manual_fusion: w*norm(clf) + (1-w)*norm(centroid); tune w on VAL ---
    clf_norm = RankNormalizer(feat["clf_score"].values[train_mask])
    ctx_norm = RankNormalizer(feat["centroid_eucl_dist"].values[train_mask])
    clf_n = clf_norm.transform(feat["clf_score"].values)
    ctx_n = ctx_norm.transform(feat["centroid_eucl_dist"].values)
    sel = cfg["selection_metric"]
    best_w, best_key = 0.5, (-np.inf, np.inf)
    for w in cfg["weight_grid"]:
        s = w * clf_n + (1 - w) * ctx_n
        ranked_val = rank_within_patient(df[val_mask], s[val_mask], cfg)
        mv = review_budget_eval(ranked_val, cfg)
        key = (mv.get(sel, -np.inf), -mv.get("mean_rank_first_malignant", np.inf))
        if key > best_key:
            best_key, best_w = key, w
    scores["manual_fusion"] = best_w * clf_n + (1 - best_w) * ctx_n
    extras["manual_fusion_weight"] = best_w

    # 5/6. learned fusion: train a model on TRAIN feature vectors ----------
    feat_cols = [c for c in feat.columns]  # use all engineered features
    Xf = feat[feat_cols].values
    fscaler = StandardScaler().fit(Xf[train_mask])  # fit on TRAIN only
    Xf_s = fscaler.transform(Xf)
    if len(np.unique(y_train)) >= 2:
        if "learned_fusion_logreg" in cfg["methods"]:
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Xf_s[train_mask], y_train)
            scores["learned_fusion_logreg"] = lr.predict_proba(Xf_s)[:, 1]
        if "learned_fusion_mlp" in cfg["methods"]:
            mlp = MLPClassifier(hidden_layer_sizes=tuple(cfg["mlp_hidden"]),
                                max_iter=400, random_state=cfg["split_seed"])
            mlp.fit(Xf_s[train_mask], y_train)
            scores["learned_fusion_mlp"] = mlp.predict_proba(Xf_s)[:, 1]
    else:
        print(">> WARNING: single-class TRAIN labels; learned fusion skipped.")

    # keep only requested + available methods, preserving config order
    scores = {m: scores[m] for m in cfg["methods"] if m in scores}
    return scores, extras, (clf_n, ctx_n)


# ----------------------------------------------------------------------------- #
# Readable report
# ----------------------------------------------------------------------------- #
def write_readable_report(path, cfg, best_method, summary_df, paired_rows, extras):
    lines = []
    A = lines.append
    A("=" * 70)
    A("EXPERIMENT 2 - PATIENT-CONTEXT FUSION RANKING - READABLE REPORT")
    A("=" * 70)
    A(f"split_seed = {cfg['split_seed']}   selection_metric = {cfg['selection_metric']}")
    A(f"manual_fusion weight selected on validation = {extras['manual_fusion_weight']}")
    A(f"best method on validation = {best_method}")
    A("")
    A("Primary test metrics by method (review-budget; lower rank is better):")
    cols = (["method"] + [f"recall@{k}" for k in cfg["topk_values"]]
            + ["mean_rank_first_malignant", "mean_percentile_rank_first_malignant"])
    A(summary_df[cols].to_string(index=False))
    A("")
    A("Does context / fusion improve over classifier_only? "
      "(paired patient bootstrap on test)")
    A("-" * 70)

    improved_any = False
    for method, rows in paired_rows.items():
        better = [r["metric"] for r in rows if r["improves_over_baseline"]]
        worse = [r["metric"] for r in rows
                 if r["ci_excludes_0"] and not r["improves_over_baseline"]]
        if better:
            improved_any = True
            A(f"* {method}: improves on {', '.join(better)} "
              f"(95% CI excludes 0 in the better direction).")
        if worse:
            A(f"    (note: {method} is significantly WORSE on {', '.join(worse)}.)")
        if not better and not worse:
            A(f"* {method}: no metric reaches significance vs classifier_only "
              "at this seed.")
    A("")
    if improved_any:
        A("CONCLUSION (this seed): context/fusion adds signal over classifier-only "
          "on at least one primary budget metric. Stability across seeds must be "
          "confirmed with aggregate_multiseed.py before drawing a firm conclusion.")
        A("This SUPPORTS continuing the patient-context ranking direction, pending "
          "the multiseed check.")
    else:
        A("CONCLUSION (this seed): no significant improvement over classifier-only. "
          "Check the multiseed aggregate; if this persists, the context-ranking "
          "direction needs rethinking rather than scaling up.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- #
# MAIN
# ----------------------------------------------------------------------------- #
def main():
    cfg = parse_args_into_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    pid, lid, lab = cfg["patient_id_col"], cfg["lesion_id_col"], cfg["label_col"]

    # --- data ------------------------------------------------------------
    df, embed_cols = load_data(cfg)
    real_data_checklist(df, embed_cols, cfg)
    audit_dataset(df, embed_cols, cfg)

    # --- split -----------------------------------------------------------
    masks = patient_disjoint_split(df, cfg)
    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]

    # --- embedding scaling (fit on TRAIN) + first-stage classifier -------
    X = df[embed_cols].values
    escaler = StandardScaler().fit(X[train_mask])
    Xs = escaler.transform(X)
    y_train = df[lab].values[train_mask]
    if len(np.unique(y_train)) < 2:
        sys.exit("ERROR: TRAIN split is single-class; cannot fit classifier.")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs[train_mask], y_train)
    clf_score = clf.predict_proba(Xs)[:, 1]

    # --- context features ------------------------------------------------
    feat = compute_context_features(df, Xs, clf_score, train_mask, cfg)
    print(f">> computed {feat.shape[1]} context features: {list(feat.columns)}\n")

    # --- method scores ---------------------------------------------------
    scores, extras, (clf_n, ctx_n) = build_method_scores(df, feat, masks, cfg)

    # --- evaluate every method on VAL (selection) and TEST (reporting) ---
    summary_rows, per_patient_tables, test_ranked = [], {}, {}
    val_select = {}
    for method, s in scores.items():
        ranked_val = rank_within_patient(df[val_mask], s[val_mask], cfg)
        val_m = review_budget_eval(ranked_val, cfg)
        val_select[method] = (val_m.get(cfg["selection_metric"], -np.inf),
                              -val_m.get("mean_rank_first_malignant", np.inf))

        ranked_test = rank_within_patient(df[test_mask], s[test_mask], cfg)
        test_m = review_budget_eval(ranked_test, cfg)
        burden = review_burden(ranked_test, cfg)
        sec = lesion_level_secondary(df[lab].values[test_mask], s[test_mask])
        row = {"method": method, "split": "test", **test_m,
               "auroc": sec["auroc"], "auprc": sec["auprc"],
               "mean_lesions_per_patient": burden["mean_lesions_per_patient"],
               "benign_only_mean_top1_score": burden["benign_only_mean_top1_score"]}
        for k in cfg["topk_values"]:
            row[f"top{k}_pct_of_lesions_reviewed"] = burden[f"top{k}_pct_of_lesions_reviewed"]
        summary_rows.append(row)
        test_ranked[method] = ranked_test
        per_patient_tables[method] = ppb.per_patient_first_ranks(ranked_test, pid, lab)

    summary_df = pd.DataFrame(summary_rows)

    # --- best method on VALIDATION (tie-break lower mean rank) ------------
    best_method = max(val_select, key=val_select.get)

    # --- paired patient bootstrap vs classifier_only (on TEST) -----------
    merged = ppb.build_merged_table(per_patient_tables, pid, lab)
    baseline = "classifier_only"
    paired_rows, paired_records = {}, []
    for method in scores:
        if method == baseline:
            continue
        rep = ppb.paired_bootstrap(
            merged, f"fr_{baseline}", f"fr_{method}", cfg["topk_values"],
            n_bootstrap=cfg["n_bootstrap"], seed=cfg["split_seed"])
        paired_rows[method] = rep
        for r in rep:
            paired_records.append({"method": method, "baseline": baseline, **r})

    # --- write outputs ---------------------------------------------------
    od = cfg["output_dir"]
    summary_df.to_csv(os.path.join(od, "summary.csv"), index=False)
    pd.DataFrame(paired_records).to_csv(
        os.path.join(od, "paired_comparison_report.csv"), index=False)

    # predictions_best.csv (best-on-val method, full per-lesion test table)
    best_ranked = test_ranked[best_method]
    idx = df.index[test_mask]
    preds = pd.DataFrame({
        "patient_id": best_ranked[pid].values,
        "lesion_id": best_ranked[lid].values,
        "true_label": best_ranked[lab].values,
        "classifier_score": clf_score[test_mask],
        "context_score": feat["centroid_eucl_dist"].values[test_mask],
        "fused_score": best_ranked["_score"].values,
        "method": best_method,
        "patient_rank": best_ranked["patient_rank"].values,
    }).sort_values(["patient_id", "patient_rank"]).reset_index(drop=True)
    preds.to_csv(os.path.join(od, "predictions_best.csv"), index=False)

    write_readable_report(os.path.join(od, "readable_report.txt"),
                          cfg, best_method, summary_df, paired_rows, extras)

    # --- console summary -------------------------------------------------
    print("=" * 70)
    print("TEST SUMMARY (primary = review-budget; AUROC/AUPRC are secondary)")
    print("=" * 70)
    show = ["method"] + [f"recall@{k}" for k in cfg["topk_values"]] + [
        "mean_rank_first_malignant", "mean_percentile_rank_first_malignant",
        "auroc", "auprc"]
    print(summary_df[show].to_string(index=False))
    print(f"\n>> best method on validation: {best_method} "
          f"(manual_fusion w={extras['manual_fusion_weight']})")
    print(f">> outputs written to {od}/  "
          "(summary.csv, predictions_best.csv, paired_comparison_report.csv, "
          "readable_report.txt)")


if __name__ == "__main__":
    main()
