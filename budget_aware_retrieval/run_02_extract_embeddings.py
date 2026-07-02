#!/usr/bin/env python3
"""run_02_extract_embeddings.py — extract & cache per-lesion embeddings (or synthetic)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import load_config
from src import embedding_extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fable5_smoke.yaml")
    ap.add_argument("--run_mode", default=None)
    ap.add_argument("--hdf5", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--use_dino", action="store_true")
    a = ap.parse_args()
    ov = {}
    if a.run_mode: ov["run_mode"] = a.run_mode
    if a.hdf5: ov.setdefault("paths", {})["hdf5"] = a.hdf5
    if a.image_dir: ov.setdefault("paths", {})["image_dir"] = a.image_dir
    if a.work_dir: ov.setdefault("paths", {})["work_dir"] = a.work_dir
    if a.results_dir: ov.setdefault("paths", {})["results_dir"] = a.results_dir
    if a.model: ov.setdefault("embedding", {})["model"] = a.model
    if a.use_dino: ov.setdefault("embedding", {})["use_dino"] = True
    cfg = load_config(a.config, ov)
    embedding_extractor.extract(cfg)


if __name__ == "__main__":
    main()
