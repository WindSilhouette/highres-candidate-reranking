"""
src/error_analysis.py
=====================
Where does rank-aware retrieval help and where does it hurt? Builds case-level
CSVs, stratified analysis, curves, and plots by comparing the best rank-aware
method's per-lesion ranking against classifier_only on a chosen seed.

Inputs are two predictions frames (best method + classifier_only) with columns:
    patient_id, lesion_id, true_label, patient_rank, and score columns.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _first_rank(df, rank_col="patient_rank"):
    """Per patient: rank of first malignant under df's ordering (NaN if none)."""
    out = {}
    for pid, g in df.groupby("patient_id"):
        mal = g[g["true_label"] == 1]
        out[pid] = int(mal[rank_col].min()) if len(mal) else np.nan
    return pd.Series(out, name="first_rank")


def case_tables(best_df, base_df, out_dir, k=5):
    """success = best moves first malignant into top-k where classifier missed;
    failure = best pushes it out of a rank the classifier had."""
    fb = _first_rank(best_df); fc = _first_rank(base_df)
    j = pd.concat([fc.rename("clf_first_rank"), fb.rename("best_first_rank")], axis=1).dropna()
    j["delta_rank"] = j["best_first_rank"] - j["clf_first_rank"]  # negative = better

    success = j[(j["clf_first_rank"] > k) & (j["best_first_rank"] <= k)].copy()
    success.sort_values("delta_rank").to_csv(os.path.join(out_dir, "top_success_cases.csv"))
    failure = j[j["delta_rank"] > 0].sort_values("delta_rank", ascending=False).copy()
    failure.to_csv(os.path.join(out_dir, "top_failure_cases.csv"))

    # benign lesions ranked above the patient's first malignant, under best method
    fb_map = fb.to_dict()
    rows = []
    for pid, g in best_df.groupby("patient_id"):
        fr = fb_map.get(pid, np.nan)
        if np.isnan(fr):
            continue
        benign_above = g[(g["true_label"] == 0) & (g["patient_rank"] < fr)]
        for _, r in benign_above.iterrows():
            rows.append({"patient_id": pid, "lesion_id": r["lesion_id"],
                         "benign_rank": int(r["patient_rank"]),
                         "first_malignant_rank": int(fr)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "false_top_benign_cases.csv"), index=False)
    return j


def stratified_analysis(best_df, base_df, meta, out_dir):
    """delta first-rank stratified by lesion-count quartile, site, age band, sex."""
    fb = _first_rank(best_df); fc = _first_rank(base_df)
    d = pd.concat([fc.rename("clf"), fb.rename("best")], axis=1).dropna()
    d["delta_rank"] = d["best"] - d["clf"]
    pat = meta.groupby("patient_id").agg(
        n_lesions=("isic_id", "size"),
        site=("anatom_site_general", lambda s: s.mode().iloc[0] if "anatom_site_general" in meta and len(s.mode()) else "NA"),
        age=("age_approx", "median") if "age_approx" in meta else ("isic_id", "size"),
        sex=("sex", lambda s: s.mode().iloc[0] if "sex" in meta and len(s.mode()) else "NA"),
    )
    d = d.join(pat, how="left")
    d["lesion_count_quartile"] = pd.qcut(d["n_lesions"], 4, labels=False, duplicates="drop")
    if "age" in d:
        d["age_band"] = pd.cut(d["age"], [0, 40, 55, 70, 200],
                               labels=["<40", "40-55", "55-70", "70+"])
    rows = []
    for by in ["lesion_count_quartile", "site", "age_band", "sex"]:
        if by not in d:
            continue
        for val, grp in d.groupby(by):
            rows.append({"stratifier": by, "value": str(val), "n_patients": len(grp),
                         "mean_delta_rank": round(grp["delta_rank"].mean(), 3),
                         "median_delta_rank": round(grp["delta_rank"].median(), 3)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "stratified_analysis.csv"), index=False)
    return d


def recall_at_k_curve(pred_by_method, out_dir, kmax=20):
    """recall@k vs k for every method, written to CSV and plotted."""
    rows = []
    for method, df in pred_by_method.items():
        fr = _first_rank(df).dropna().to_numpy()
        for k in range(1, kmax + 1):
            rows.append({"method": method, "k": k, "recall_at_k": float(np.mean(fr <= k))})
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(out_dir, "recall_at_k_curve.csv"), index=False)
    plt.figure(figsize=(7, 5))
    for method, g in curve.groupby("method"):
        plt.plot(g["k"], g["recall_at_k"], marker="o", ms=3, label=method)
    plt.xlabel("review budget k"); plt.ylabel("recall@k (patient-level)")
    plt.title("Recall@k under review budget"); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "recall_at_k_curve.png"), dpi=120)
    plt.close()
    return curve


def distribution_plots(best_df, base_df, joined, out_dir, best_name):
    # delta-rank distribution
    joined["delta_rank"].to_csv(os.path.join(out_dir, "delta_rank_distribution.csv"))
    plt.figure(figsize=(7, 4))
    plt.hist(joined["delta_rank"], bins=30, color="#4472c4")
    plt.axvline(0, color="k", lw=1)
    plt.xlabel(f"delta first-malignant rank ({best_name} - classifier_only), <0 = better")
    plt.ylabel("patients"); plt.title("Delta-rank distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "delta_rank_hist.png"), dpi=120); plt.close()

    # first-rank histograms
    plt.figure(figsize=(7, 4))
    plt.hist(_first_rank(base_df).dropna(), bins=30, alpha=0.6, label="classifier_only")
    plt.hist(_first_rank(best_df).dropna(), bins=30, alpha=0.6, label=best_name)
    plt.xlabel("first-malignant rank"); plt.ylabel("patients")
    plt.legend(); plt.title("First-malignant rank distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "rank_distribution_hist.png"), dpi=120); plt.close()

    # classifier vs context scatter
    if {"classifier_score", "context_score"}.issubset(best_df.columns):
        plt.figure(figsize=(6, 5))
        m = best_df["true_label"] == 1
        plt.scatter(best_df.loc[~m, "classifier_score"], best_df.loc[~m, "context_score"],
                    s=5, alpha=0.3, label="benign")
        plt.scatter(best_df.loc[m, "classifier_score"], best_df.loc[m, "context_score"],
                    s=14, color="crimson", label="malignant")
        plt.xlabel("classifier score"); plt.ylabel("context score")
        plt.legend(); plt.title("Classifier vs context")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "clf_vs_context_scatter.png"), dpi=120); plt.close()


def method_comparison_bar(summary_df, out_dir):
    metrics = ["recall@5", "recall@10", "mean_rank_first_malignant"]
    metrics = [m for m in metrics if m in summary_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, met in zip(axes, metrics):
        s = summary_df.sort_values(met, ascending=(met == "mean_rank_first_malignant"))
        ax.barh(s["method"], s[met], color="#4472c4")
        ax.set_title(met); ax.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "method_comparison_bar.png"), dpi=120); plt.close()
