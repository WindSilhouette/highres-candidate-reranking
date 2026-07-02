#!/usr/bin/env python3
"""
run_06_error_report.py
======================
Build the error-analysis CSVs and plots for a chosen seed: where the best
rank-aware method helps vs classifier_only, stratified breakdowns, and the
recall@k curve across methods.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.io_utils import read_table

from src.config import load_config, artifact_paths
from src import error_analysis as ea


def _ranks_to_pred(ranks_table, method):
    col = f"rank_{method}"
    if col not in ranks_table.columns:
        return None
    return ranks_table[["patient_id", "lesion_id", "true_label",
                        "classifier_score", "context_score"]].assign(
        patient_rank=ranks_table[col].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fable5_smoke.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_mode", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    a = ap.parse_args()
    ov = {"run_mode": a.run_mode} if a.run_mode else {}
    if a.work_dir: ov.setdefault("paths", {})["work_dir"] = a.work_dir
    if a.results_dir: ov.setdefault("paths", {})["results_dir"] = a.results_dir
    cfg = load_config(a.config, ov)
    rd = cfg["paths"]["results_dir"]
    seed_dir = os.path.join(rd, f"seed_{a.seed}")
    err_dir = os.path.join(rd, "error"); plot_dir = os.path.join(rd, "plots")
    os.makedirs(err_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    ranks = read_table(os.path.join(seed_dir, "all_method_ranks.parquet"))
    with open(os.path.join(seed_dir, "seed_meta.json")) as f:
        best_method = json.load(f)["best_method"]
    summary = pd.read_csv(os.path.join(seed_dir, "summary.csv"))

    # Focus on the validation-selected best if it is rank-aware. If the overall
    # best is classifier_only/metadata/context, choose the strongest available
    # rank-aware model for the case study instead of silently taking the first
    # method in a hard-coded list. This keeps error analysis aligned with the
    # actual result table.
    available = [c[5:] for c in ranks.columns if c.startswith("rank_")]
    rank_aware = [m for m in ["pairwise_rank_logreg", "lambda_pairwise_logreg",
                              "pairwise_rank_mlp", "listwise_softmax_ranker"]
                  if m in available]
    if best_method in rank_aware:
        focus = best_method
    elif rank_aware:
        sub = summary[summary["method"].isin(rank_aware)].copy()
        # Test-table sorting is for visualization only; model selection remains
        # the validation decision stored in seed_meta.json.
        sort_cols = [c for c in ["recall@5", "mean_rank_first_malignant", "recall@10"] if c in sub.columns]
        ascending = [False if c != "mean_rank_first_malignant" else True for c in sort_cols]
        focus = sub.sort_values(sort_cols, ascending=ascending).iloc[0]["method"] if len(sub) else best_method
    else:
        focus = best_method if best_method in available else "classifier_only"
    print(f">> error analysis: focus method = {focus} (validation-selected best = {best_method}, seed {a.seed})")

    best_df = _ranks_to_pred(ranks, focus)
    base_df = _ranks_to_pred(ranks, "classifier_only")
    if best_df is None or base_df is None:
        raise SystemExit(f"[FATAL] could not build rank tables for focus={focus} and classifier_only")
    meta = read_table(artifact_paths(cfg)["metadata"])

    joined = ea.case_tables(best_df, base_df, err_dir, k=5)
    ea.stratified_analysis(best_df, base_df, meta, err_dir)

    pred_by_method = {}
    for c in ranks.columns:
        if c.startswith("rank_"):
            pred_by_method[c[5:]] = _ranks_to_pred(ranks, c[5:])
    ea.recall_at_k_curve(pred_by_method, err_dir, kmax=max(cfg["eval"]["topk_values"]))
    ea.distribution_plots(best_df, base_df, joined, plot_dir, focus)

    ea.method_comparison_bar(summary, plot_dir)
    # move the curve png next to the other plots too
    import shutil
    if os.path.exists(os.path.join(err_dir, "recall_at_k_curve.png")):
        shutil.copy(os.path.join(err_dir, "recall_at_k_curve.png"),
                    os.path.join(plot_dir, "recall_at_k_curve.png"))
    print(f">> error CSVs -> {err_dir}/  plots -> {plot_dir}/")


if __name__ == "__main__":
    main()
