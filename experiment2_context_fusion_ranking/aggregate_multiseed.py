#!/usr/bin/env python3
"""
aggregate_multiseed.py
======================
Aggregate per-seed Experiment 2 outputs into multiseed mean/std tables, so we
can judge whether any improvement of context/fusion over classifier_only is
STABLE rather than a single-seed artefact.

Reads:   <results_dir>/seed_*/summary.csv
         <results_dir>/seed_*/paired_comparison_report.csv
Writes:  <results_dir>/multiseed_mean_std.csv
         <results_dir>/multiseed_paired_report.csv

Usage:   python aggregate_multiseed.py --results_dir results
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def _seed_from_path(p):
    base = os.path.basename(os.path.dirname(p))  # seed_42
    try:
        return int(base.split("_")[-1])
    except ValueError:
        return -1


def aggregate_summary(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "seed_*", "summary.csv")))
    if not files:
        return None, 0
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["seed"] = _seed_from_path(f)
        frames.append(d)
    alld = pd.concat(frames, ignore_index=True)
    num_cols = [c for c in alld.columns
                if c not in ("method", "split", "seed")
                and pd.api.types.is_numeric_dtype(alld[c])]
    agg = (alld.groupby("method")[num_cols]
           .agg(["mean", "std"]).reset_index())
    # flatten the MultiIndex columns -> "<metric>_mean" / "<metric>_std"
    agg.columns = ["method"] + [f"{m}_{s}" for m, s in agg.columns[1:]]
    return agg, alld["seed"].nunique()


def aggregate_paired(results_dir):
    files = sorted(glob.glob(
        os.path.join(results_dir, "seed_*", "paired_comparison_report.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["seed"] = _seed_from_path(f)
        frames.append(d)
    alld = pd.concat(frames, ignore_index=True)
    # per (method, metric): mean/std delta, mean CI bounds, and how often the
    # CI excluded 0 in the better direction across seeds (stability signal)
    grp = alld.groupby(["method", "metric"])
    out = grp.agg(
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
        ci_low_mean=("ci_low", "mean"),
        ci_high_mean=("ci_high", "mean"),
        n_seeds=("seed", "nunique"),
        frac_seeds_ci_excludes_0=("ci_excludes_0", "mean"),
        frac_seeds_improves=("improves_over_baseline", "mean"),
    ).reset_index()
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate Experiment 2 seeds.")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    summary, n_seeds = aggregate_summary(args.results_dir)
    if summary is None:
        print(f"No seed_*/summary.csv found under {args.results_dir}/. "
              "Run the per-seed experiment first.")
        return
    summary.to_csv(os.path.join(args.results_dir, "multiseed_mean_std.csv"), index=False)

    paired = aggregate_paired(args.results_dir)
    if paired is not None:
        paired.to_csv(os.path.join(args.results_dir, "multiseed_paired_report.csv"),
                      index=False)

    print(f"Aggregated {n_seeds} seed(s).")
    print(f"-> {args.results_dir}/multiseed_mean_std.csv")
    if paired is not None:
        print(f"-> {args.results_dir}/multiseed_paired_report.csv\n")
        # short stability read-out: which method/metric improves on a majority of seeds
        stable = paired[(paired["frac_seeds_improves"] >= 0.5)
                        & (paired["n_seeds"] >= 2)]
        if len(stable):
            print("Stable improvements over classifier_only "
                  "(better-direction CI excludes 0 on >=50% of seeds):")
            for _, r in stable.iterrows():
                print(f"  {r['method']:24s} {r['metric']:32s} "
                      f"mean delta={r['delta_mean']:+.3f}  "
                      f"({r['frac_seeds_improves']*100:.0f}% of seeds)")
        else:
            print("No method shows a stable (>=50% of seeds) improvement over "
                  "classifier_only. Treat the context-ranking gain as unconfirmed.")


if __name__ == "__main__":
    main()
