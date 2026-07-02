"""
src/bootstrap.py
================
Paired, patient-level bootstrap: compare each method against classifier_only by
resampling PATIENTS (not lesions) with replacement, using the SAME resampled set
for both methods so the delta isolates the method effect.

Operates on per-patient "first malignant rank" tables, which is all the
review-budget metrics need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# metric -> +1 if higher is better, -1 if lower is better
DELTA_METRICS = {
    "recall@1": +1, "recall@3": +1, "recall@5": +1, "recall@10": +1, "recall@20": +1,
    "mean_rank_first_malignant": -1, "NNR": -1, "normalized_percentile_rank": -1,
}


def _metric(fr, nl, name):
    fr = np.asarray(fr, float); nl = np.asarray(nl, float)
    if fr.size == 0:
        return float("nan")
    if name.startswith("recall@"):
        return float(np.mean(fr <= int(name.split("@")[1])))
    if name in ("mean_rank_first_malignant", "NNR"):
        return float(np.mean(fr))
    if name == "normalized_percentile_rank":
        return float(np.mean(fr / nl))
    raise ValueError(name)


def build_merged(per_method_tables):
    """Merge {method: per_patient_first_rank_table} into one aligned frame."""
    methods = list(per_method_tables)
    base = per_method_tables[methods[0]][["n_lesions", "malignant_patient"]].copy()
    for m in methods:
        base[f"fr_{m}"] = per_method_tables[m]["first_rank"]
    return base


def paired_bootstrap(merged, baseline_col, method_col, metrics,
                     n_bootstrap=1000, seed=0):
    """Return one dict per metric: point delta, 95% CI, CI-excludes-0, improves."""
    rng = np.random.default_rng(seed)
    mal = merged[merged["malignant_patient"] == 1]
    nl = mal["n_lesions"].to_numpy()
    fb = mal[baseline_col].to_numpy(); fm = mal[method_col].to_numpy()
    n = len(mal)
    out = []
    for name in metrics:
        point = _metric(fm, nl, name) - _metric(fb, nl, name)
        deltas = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            deltas[b] = _metric(fm[idx], nl[idx], name) - _metric(fb[idx], nl[idx], name)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        excl = bool(lo > 0 or hi < 0)
        improves = bool(excl and np.sign(point) == DELTA_METRICS.get(name, +1))
        out.append({"metric": name, "delta": round(float(point), 4),
                    "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                    "ci_excludes_0": excl, "improves_over_baseline": improves,
                    "n_malignant_patients": int(n)})
    return out


def bootstrap_metrics_list(topk_values):
    ks = [f"recall@{k}" for k in topk_values]
    return ks + ["mean_rank_first_malignant", "NNR", "normalized_percentile_rank"]
