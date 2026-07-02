"""
src/evaluator.py
================
Budget-aware retrieval metrics. PRIMARY = how early the first malignant lesion
appears in a patient's within-patient ranking (top-k / rank / NNR / percentile).
SECONDARY = lesion-level AUROC / AUPRC / partial-AUC.

All ranking is WITHIN patient: we rank each patient's own lesions by a score and
ask where that patient's first malignant lesion lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


PID, LAB, RANK = "patient_id", "malignant", "patient_rank"


def rank_within_patient(df, scores):
    """Attach a 1-indexed within-patient rank (1 = reviewed first)."""
    out = df.copy()
    out["_score"] = np.asarray(scores, float)
    out[RANK] = (out.groupby(PID)["_score"]
                 .rank(method="first", ascending=False).astype(int))
    return out


def per_patient_first_rank(ranked):
    """One row per patient: n_lesions, malignant_patient, first_rank (NaN benign)."""
    rows = []
    for pid, g in ranked.groupby(PID):
        has = int(g[LAB].sum() > 0)
        fr = int(g.loc[g[LAB] == 1, RANK].min()) if has else np.nan
        rows.append((pid, len(g), has, fr))
    return pd.DataFrame(rows, columns=[PID, "n_lesions", "malignant_patient",
                                       "first_rank"]).set_index(PID)


def budget_metrics(ranked, topk_values, top_pct=0.10):
    """Primary review-budget metrics over malignant-positive patients."""
    first, pctile, top_pct_hit = [], [], []
    hits = {k: [] for k in topk_values}
    for _, g in ranked.groupby(PID):
        if g[LAB].sum() == 0:
            continue
        n = len(g)
        r = int(g.loc[g[LAB] == 1, RANK].min())
        first.append(r)
        pctile.append(r / n)
        for k in topk_values:
            hits[k].append(1 if r <= k else 0)
        budget = max(1, int(np.ceil(top_pct * n)))     # review top X% of this patient
        top_pct_hit.append(1 if r <= budget else 0)
    first = np.array(first, float)
    m = {f"recall@{k}": float(np.mean(hits[k])) if hits[k] else float("nan")
         for k in topk_values}
    m["mean_rank_first_malignant"] = float(first.mean()) if first.size else float("nan")
    m["median_rank_first_malignant"] = float(np.median(first)) if first.size else float("nan")
    m["NNR"] = m["mean_rank_first_malignant"]          # number-needed-to-review == mean rank
    m["normalized_percentile_rank"] = float(np.mean(pctile)) if pctile else float("nan")
    m[f"recall_top{int(top_pct*100)}pct"] = float(np.mean(top_pct_hit)) if top_pct_hit else float("nan")
    m["n_malignant_patients_evaluated"] = int(first.size)
    return m


def benign_burden(ranked):
    """Benign-only patient review burden: mean score of their top-ranked lesion."""
    tops = [float(g.loc[g[RANK] == 1, "_score"].iloc[0])
            for _, g in ranked.groupby(PID) if g[LAB].sum() == 0]
    return {"benign_only_mean_top1_score": float(np.mean(tops)) if tops else float("nan"),
            "n_benign_only_patients": len(tops)}


def partial_auc_above_tpr(y, scores, tpr_min=0.80):
    """ISIC-style standardized partial AUC for the high-sensitivity region.

    ISIC 2024 evaluates the area associated with TPR >= ``tpr_min`` by flipping
    labels/scores and using sklearn's standardized partial AUC with
    ``max_fpr = 1 - tpr_min``. This returns 0.5 for random and 1.0 for perfect,
    avoiding the previous incorrect integration over an arbitrary ROC slice.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    max_fpr = float(np.clip(1.0 - tpr_min, 1e-6, 1.0))
    # Flip the problem to match the public ISIC min_tpr convention.
    return float(roc_auc_score(1 - y, -scores, max_fpr=max_fpr))


def secondary_metrics(y, scores, pauc_tpr_min=0.80):
    y = np.asarray(y); scores = np.asarray(scores, float)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "pauc": float("nan")}
    return {"auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)),
            "pauc": partial_auc_above_tpr(y, scores, pauc_tpr_min)}


def evaluate(df_test, scores, cfg):
    """Full metric bundle for one method's test scores."""
    ranked = rank_within_patient(df_test, scores)
    m = budget_metrics(ranked, cfg["eval"]["topk_values"], cfg["eval"]["top_pct"])
    m.update(benign_burden(ranked))
    m.update(secondary_metrics(df_test[LAB].values, scores, cfg["eval"]["pauc_tpr_min"]))
    return m, ranked


def selection_key(m, cfg):
    """(primary up, secondary down, tertiary up) -> maximise this tuple on val."""
    e = cfg["eval"]
    return (m.get(e["selection_primary"], -np.inf),
            -m.get(e["selection_secondary"], np.inf),
            m.get(e["selection_tertiary"], -np.inf))
