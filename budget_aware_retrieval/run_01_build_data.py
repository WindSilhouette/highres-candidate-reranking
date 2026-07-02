#!/usr/bin/env python3
"""run_01_build_data.py — load/audit ISIC metadata and write patient-disjoint splits."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import load_config
from src import data_builder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fable5_smoke.yaml")
    ap.add_argument("--run_mode", default=None)
    ap.add_argument("--metadata_csv", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--patient_col", default=None)
    ap.add_argument("--target_col", default=None)
    ap.add_argument("--image_path_col", default=None)
    ap.add_argument("--max_lesions", type=int, default=None)
    ap.add_argument("--max_patients", type=int, default=None)
    a = ap.parse_args()
    ov = {}
    if a.run_mode: ov["run_mode"] = a.run_mode
    if a.metadata_csv: ov.setdefault("paths", {})["metadata_csv"] = a.metadata_csv
    if a.work_dir: ov.setdefault("paths", {})["work_dir"] = a.work_dir
    if a.results_dir: ov.setdefault("paths", {})["results_dir"] = a.results_dir
    for key in ["id_col", "patient_col", "target_col", "image_path_col"]:
        val = getattr(a, key)
        if val: ov.setdefault("data", {})[key] = val
    if a.max_lesions: ov.setdefault("subset", {})["max_lesions"] = a.max_lesions
    if a.max_patients: ov.setdefault("subset", {})["max_patients"] = a.max_patients
    cfg = load_config(a.config, ov)
    data_builder.build(cfg)


if __name__ == "__main__":
    main()
