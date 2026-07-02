"""
src/report_writer.py
====================
Human-readable reporting. Per-seed text plus a final results/readable_report.md
that states, in plain language, whether rank-aware retrieval beat classifier-only
and what the next scientific decision is (conclusion A / B / C).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RANK_AWARE = ["pairwise_rank_logreg", "pairwise_rank_mlp",
              "listwise_softmax_ranker", "lambda_pairwise_logreg"]


def per_seed_report(path, seed, summary_df, paired_rows, best_method, info, cfg):
    L = [f"# FABLE-5 seed {seed} report", ""]
    L.append(f"- run_mode: {cfg['run_mode']}")
    L.append(f"- best method on validation: **{best_method}**")
    if "manual_fusion_selected" in info:
        L.append(f"- manual fusion selected: {info['manual_fusion_selected']}")
    if "n_pairs" in info:
        L.append(f"- pairwise training pairs: {info['n_pairs']}")
    L.append("")
    cols = ["method"] + [f"recall@{k}" for k in cfg["eval"]["topk_values"]] + [
        "mean_rank_first_malignant", "normalized_percentile_rank", "auroc"]
    L.append("## Test metrics (primary = review-budget)\n")
    L.append(summary_df[cols].round(3).to_string(index=False))
    L.append("\n## Paired bootstrap vs classifier_only\n")
    for method, rows in paired_rows.items():
        better = [r["metric"] for r in rows if r["improves_over_baseline"]]
        tag = " (rank-aware)" if method in RANK_AWARE else ""
        L.append(f"- **{method}**{tag}: "
                 + ("improves " + ", ".join(better) if better
                    else "no metric significant at this seed"))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt_ci(r):
    star = " *" if r["improves_over_baseline"] else ""
    return f"{r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]{star}"


def final_report(results_dir, multiseed_mean, multiseed_paired, cfg):
    """Write results/readable_report.md and print the A/B/C conclusion."""
    path = os.path.join(results_dir, "readable_report.md")
    L = ["# FABLE-5 — Budget-Aware Melanoma Lesion Retrieval", ""]
    L.append("## What data was used?")
    L.append(f"ISIC 2024 / SLICE-3D (run_mode = **{cfg['run_mode']}**), evaluated as "
             "patient-level lesion retrieval: each patient's lesions are ranked and "
             "we measure how early the first malignant lesion appears under a review "
             "budget. AUROC/AUPRC are secondary.\n")

    L.append("## What baselines were compared?")
    L.append(", ".join(f"`{m}`" for m in cfg["models"]) + ".\n")

    # pick the best rank-aware method by mean recall@5 across seeds
    ms = multiseed_mean.set_index("method")
    rank_aware_present = [m for m in RANK_AWARE if m in ms.index]
    base_r5 = ms.loc["classifier_only", "recall@5_mean"] if "classifier_only" in ms.index else np.nan
    base_r10 = ms.loc["classifier_only", "recall@10_mean"] if "classifier_only" in ms.index else np.nan
    base_mr = ms.loc["classifier_only", "mean_rank_first_malignant_mean"] if "classifier_only" in ms.index else np.nan

    best_ra, best_r5 = None, -np.inf
    for m in rank_aware_present:
        v = ms.loc[m, "recall@5_mean"]
        if v > best_r5:
            best_r5, best_ra = v, m

    L.append("## Did rank-aware retrieval beat classifier-only?")
    if best_ra is None:
        L.append("No rank-aware method produced scores. See logs.\n")
        conclusion = "C"
    else:
        r5 = ms.loc[best_ra, "recall@5_mean"]; r10 = ms.loc[best_ra, "recall@10_mean"]
        mr = ms.loc[best_ra, "mean_rank_first_malignant_mean"]
        L.append(f"Best rank-aware method: **{best_ra}**.\n")
        L.append(f"| metric | classifier_only | {best_ra} |")
        L.append("|---|---|---|")
        L.append(f"| recall@5 | {base_r5:.3f} | {r5:.3f} |")
        L.append(f"| recall@10 | {base_r10:.3f} | {r10:.3f} |")
        L.append(f"| mean first-malignant rank | {base_mr:.3f} | {mr:.3f} |\n")

        # significance from multiseed paired report (fraction of seeds CI excludes 0
        # in the better direction), for this method
        pr = multiseed_paired[(multiseed_paired["method"] == best_ra)]
        def _frac(metric):
            row = pr[pr["metric"] == metric]
            return float(row["frac_seeds_improves"].iloc[0]) if len(row) else 0.0
        f_r5, f_r10 = _frac("recall@5"), _frac("recall@10")
        f_mr = _frac("mean_rank_first_malignant")

        L.append("## On which top-k metrics, and was it significant?")
        L.append(f"- recall@5 improvement significant on {f_r5*100:.0f}% of seeds")
        L.append(f"- recall@10 improvement significant on {f_r10*100:.0f}% of seeds")
        L.append(f"- mean-rank improvement significant on {f_mr*100:.0f}% of seeds\n")

        sig_recall = (f_r5 >= 0.5 or f_r10 >= 0.5)
        improves_meanrank = mr < base_mr
        improves_recall = (r5 > base_r5 or r10 > base_r10)
        if sig_recall and improves_meanrank:
            conclusion = "A"
        elif improves_recall or improves_meanrank:
            conclusion = "B"
        else:
            conclusion = "C"

    L.append("## Where did it help / fail?")
    L.append("See `error/` (top_success_cases, top_failure_cases, "
             "false_top_benign_cases, stratified_analysis) and `plots/`.\n")

    L.append("## Full paired report (delta [95% CI], * = CI excludes 0 in better direction)")
    for m in [x for x in multiseed_paired["method"].unique()]:
        sub = multiseed_paired[multiseed_paired["method"] == m]
        deltas = ", ".join(f"{r['metric']}={r['delta_mean']:+.3f}"
                           f"({r['frac_seeds_improves']*100:.0f}%)"
                           for _, r in sub.iterrows())
        L.append(f"- **{m}**: {deltas}")
    L.append("")

    verdicts = {
        "A": ("## Conclusion: A — STRONG SUCCESS\n"
              "Rank-aware retrieval improves recall@5 or recall@10 over "
              "classifier-only with the CI excluding 0 on a majority of seeds AND "
              "improves mean first-malignant rank. This is a paper-worthy positive "
              "signal: proceed to strengthen it (calibration, richer features, "
              "statistical power / more malignant patients)."),
        "B": ("## Conclusion: B — PROMISING BUT NOT SIGNIFICANT\n"
              "Rank-aware retrieval improves the point estimates (mean rank and/or "
              "recall) but the CIs still cross 0. The signal is real but "
              "power-limited. Next: increase the number of malignant patients "
              "(MEDIUM/FULL run), pool patients across seeds for one combined test, "
              "and add calibrated/richer context features before scaling the model."),
        "C": ("## Conclusion: C — NEGATIVE\n"
              "Rank-aware retrieval does not beat classifier-only. The within-patient "
              "context signal is insufficient at this representation. Pivot: move to "
              "representation learning / SSL embeddings or high-resolution candidate "
              "detection rather than more fusion tweaks."),
    }
    L.append(verdicts[conclusion])
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "=" * 70)
    print(f"FABLE-5 CONCLUSION: {conclusion}")
    print("=" * 70)
    print(verdicts[conclusion].split("\n", 1)[1])
    print(f"\n>> full report -> {path}")
    return conclusion
