#!/usr/bin/env python3
"""
paired_patient_bootstrap.py
===========================
Paired, patient-level bootstrap for comparing a baseline ranking method
(``classifier_only``) against each context / fusion method on TEST patients.

Why paired + patient-level:
  * patient-level  -> we resample PATIENTS, never lesions, because the unit of
    clinical review is a patient and lesions within a patient are not
    independent.
  * paired         -> in each bootstrap iteration the SAME resampled patient set
    is scored under both methods, so the delta isolates the method effect rather
    than the luck of which patients were drawn.

This module is intentionally self-contained (no import from the main script) so
it can be reused or unit-tested in isolation. It operates on a tidy per-patient
table of "rank of first malignant lesion", which is all the primary
review-budget metrics need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metrics whose delta we report. Direction = +1 if higher-is-better, else -1.
DELTA_METRICS = {
    "recall@1": +1,
    "recall@3": +1,
    "recall@5": +1,
    "recall@10": +1,
    "mean_rank_first_malignant": -1,
    "normalized_percentile_rank": -1,
}


# --------------------------------------------------------------------------- #
# Build the per-patient "first malignant rank" table from a ranked frame
# --------------------------------------------------------------------------- #
def per_patient_first_ranks(ranked_df, patient_id_col, label_col,
                            rank_col="patient_rank"):
    """Collapse a per-lesion ranked frame to one row per patient.

    Returns DataFrame indexed by patient_id with columns:
      n_lesions         : lesions for that patient
      malignant_patient : 1 if the patient has >=1 malignant lesion
      first_rank        : within-patient rank of the earliest malignant lesion
                          (NaN for benign-only patients)
    """
    rows = []
    for pid, g in ranked_df.groupby(patient_id_col):
        has_mal = int(g[label_col].sum() > 0)
        first_rank = (int(g.loc[g[label_col] == 1, rank_col].min())
                      if has_mal else np.nan)
        rows.append((pid, len(g), has_mal, first_rank))
    out = pd.DataFrame(rows, columns=[patient_id_col, "n_lesions",
                                      "malignant_patient", "first_rank"])
    return out.set_index(patient_id_col)


def _metric_value(first_ranks, n_lesions, metric, topk_values):
    """Compute a single scalar metric from arrays over MALIGNANT patients."""
    fr = np.asarray(first_ranks, dtype=float)
    nl = np.asarray(n_lesions, dtype=float)
    if fr.size == 0:
        return float("nan")
    if metric.startswith("recall@"):
        k = int(metric.split("@")[1])
        return float(np.mean(fr <= k))
    if metric == "mean_rank_first_malignant":
        return float(np.mean(fr))
    if metric == "normalized_percentile_rank":
        return float(np.mean(fr / nl))
    raise ValueError(f"unknown metric '{metric}'")


# --------------------------------------------------------------------------- #
# Paired bootstrap of (method - baseline) deltas
# --------------------------------------------------------------------------- #
def paired_bootstrap(merged, baseline_col, method_col, topk_values,
                     n_bootstrap=1000, seed=0, metrics=None):
    """Paired patient-level bootstrap of metric deltas (method - baseline).

    ``merged`` is one row per patient and must contain:
      n_lesions, malignant_patient, <baseline_col>, <method_col>
    where the *_col values are that patient's first-malignant rank under each
    method (NaN for benign-only patients, which are excluded here because the
    review-budget metrics are defined over malignant patients).

    Returns a list of dict rows, one per metric, with the point-estimate delta,
    a 95% bootstrap CI, and whether the CI excludes 0.
    """
    if metrics is None:
        metrics = list(DELTA_METRICS.keys())
    rng = np.random.default_rng(seed)

    mal = merged[merged["malignant_patient"] == 1]
    nl = mal["n_lesions"].to_numpy()
    fr_base = mal[baseline_col].to_numpy()
    fr_meth = mal[method_col].to_numpy()
    n = len(mal)

    results = []
    for metric in metrics:
        # point estimate on the full test set
        point = (_metric_value(fr_meth, nl, metric, topk_values)
                 - _metric_value(fr_base, nl, metric, topk_values))
        # bootstrap distribution of the paired delta
        deltas = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)  # same patients for both methods
            d_meth = _metric_value(fr_meth[idx], nl[idx], metric, topk_values)
            d_base = _metric_value(fr_base[idx], nl[idx], metric, topk_values)
            deltas[b] = d_meth - d_base
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        excludes_zero = bool(lo > 0 or hi < 0)
        # is the (significant) effect in the better direction for this metric?
        direction = DELTA_METRICS.get(metric, +1)
        improves = bool(excludes_zero and np.sign(point) == direction)
        results.append({
            "metric": metric,
            "delta": round(float(point), 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
            "ci_excludes_0": excludes_zero,
            "improves_over_baseline": improves,
            "n_malignant_patients": int(n),
        })
    return results


def build_merged_table(per_method_tables, patient_id_col, label_col):
    """Merge several methods' per-patient first-rank tables into one frame.

    ``per_method_tables`` : dict {method_name: per_patient_first_ranks(...)}.
    All tables share the same patient index (same test patients), so we take
    n_lesions / malignant_patient from any one of them and add a first-rank
    column per method named ``fr_<method>``.
    """
    methods = list(per_method_tables)
    base = per_method_tables[methods[0]][["n_lesions", "malignant_patient"]].copy()
    for m in methods:
        base[f"fr_{m}"] = per_method_tables[m]["first_rank"]
    return base


if __name__ == "__main__":
    # Tiny self-test so the module is runnable in isolation.
    demo = pd.DataFrame({
        "patient_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
        "malignant":  [0,   1,   0,   1,   0,   0,   0,   1],
        "patient_rank":[2,  1,   3,   1,   2,   3,   2,   1],
    })
    t = per_patient_first_ranks(demo, "patient_id", "malignant")
    print(t)
    merged = build_merged_table({"classifier_only": t, "context_only": t},
                                "patient_id", "malignant")
    rep = paired_bootstrap(merged, "fr_classifier_only", "fr_context_only",
                           [1, 3, 5, 10], n_bootstrap=100, seed=1)
    print(pd.DataFrame(rep))
