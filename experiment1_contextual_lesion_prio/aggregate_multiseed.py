#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_multiseed.py
=================================================================================
Aggregate the per-seed summary tables produced by run_multiseed.sh.

Reads every  results/seed_*/summary.csv , keeps only the TEST-split rows, and
aggregates per method (mean / std / count) over the primary triage metrics:

    recall@1, recall@3, recall@5,
    mean_rank_first_malignant,
    mean_percentile_rank_first_malignant,   (lesion-count-normalized rank proxy)
    mean_NNR_to_first_malignant

Writes:
    results/multiseed_all_test_rows.csv   (every test row, all seeds, long form)
    results/multiseed_mean_std.csv        (method x metric, mean/std/count)
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "mean_rank_first_malignant",
    "mean_percentile_rank_first_malignant",
    "mean_NNR_to_first_malignant",
]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    paths = sorted(glob.glob("results/seed_*/summary.csv"))
    if not paths:
        sys.exit("[FATAL] no results/seed_*/summary.csv found. Run ./run_multiseed.sh first.")

    frames = []
    for p in paths:
        m = re.search(r"seed_([^/\\]+)", p)
        seed = m.group(1) if m else "unknown"
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    if "split" not in all_rows.columns:
        sys.exit("[FATAL] summary files have no 'split' column — unexpected format.")
    test = all_rows[all_rows["split"] == "test"].copy()
    test.to_csv("results/multiseed_all_test_rows.csv", index=False)

    present = [m for m in METRICS if m in test.columns]
    missing = [m for m in METRICS if m not in test.columns]
    if missing:
        print(f"[aggregate] note: metrics not found in summaries and skipped: {missing}")

    agg = test.groupby("method")[present].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg.to_csv("results/multiseed_mean_std.csv", index=False)

    seeds = sorted(test["seed"].unique().tolist())
    print("=" * 70)
    print("aggregate_multiseed.py")
    print("=" * 70)
    print(f"  seeds aggregated : {seeds}")
    print(f"  test rows        : {len(test)}")
    print(f"  methods          : {test['method'].nunique()}")
    print(f"  wrote            : results/multiseed_all_test_rows.csv")
    print(f"  wrote            : results/multiseed_mean_std.csv")
    print("-" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        show = agg.copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
