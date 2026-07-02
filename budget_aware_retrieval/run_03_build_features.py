#!/usr/bin/env python3
"""run_03_build_features.py — build the static (group B + C) feature table."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import load_config
from src import feature_builder


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
    feature_builder.build_static(cfg)


if __name__ == "__main__":
    main()
