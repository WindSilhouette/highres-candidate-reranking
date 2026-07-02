#!/usr/bin/env python3
"""
run_05_multiseed.py
===================
Run train/eval across all configured seeds, aggregate to mean/std and a paired
stability report, and write results/readable_report.md with the A/B/C conclusion.
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.config import load_config
from src import report_writer as rw
from run_04_train_eval import train_eval_seed


def _seed_of(path):
    try:
        return int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    except ValueError:
        return -1


def aggregate(results_dir):
    sfiles = sorted(glob.glob(os.path.join(results_dir, "seed_*", "summary.csv")))
    frames = []
    for f in sfiles:
        d = pd.read_csv(f); d["seed"] = _seed_of(f); frames.append(d)
    alls = pd.concat(frames, ignore_index=True)
    num = [c for c in alls.columns if c not in ("method", "split", "seed")
           and pd.api.types.is_numeric_dtype(alls[c])]
    mean_std = alls.groupby("method")[num].agg(["mean", "std"]).reset_index()
    mean_std.columns = ["method"] + [f"{m}_{s}" for m, s in mean_std.columns[1:]]
    mean_std.to_csv(os.path.join(results_dir, "multiseed_mean_std.csv"), index=False)

    pfiles = sorted(glob.glob(os.path.join(results_dir, "seed_*", "paired_comparison_report.csv")))
    pf = pd.concat([pd.read_csv(f).assign(seed=_seed_of(f)) for f in pfiles], ignore_index=True)
    paired = pf.groupby(["method", "metric"]).agg(
        delta_mean=("delta", "mean"), delta_std=("delta", "std"),
        n_seeds=("seed", "nunique"),
        frac_seeds_ci_excludes_0=("ci_excludes_0", "mean"),
        frac_seeds_improves=("improves_over_baseline", "mean")).reset_index()
    paired.to_csv(os.path.join(results_dir, "multiseed_paired_report.csv"), index=False)
    return mean_std, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fable5_smoke.yaml")
    ap.add_argument("--run_mode", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    a = ap.parse_args()
    ov = {"run_mode": a.run_mode} if a.run_mode else {}
    if a.work_dir: ov.setdefault("paths", {})["work_dir"] = a.work_dir
    if a.results_dir: ov.setdefault("paths", {})["results_dir"] = a.results_dir
    cfg = load_config(a.config, ov)
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    for seed in cfg["seeds"]:
        print(f"\n########## SEED {seed} ##########")
        train_eval_seed(cfg, seed)

    mean_std, paired = aggregate(results_dir)
    rw.final_report(results_dir, mean_std, paired, cfg)
    print(f">> multiseed tables + readable_report.md -> {results_dir}/")


if __name__ == "__main__":
    main()
