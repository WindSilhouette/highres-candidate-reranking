"""
src/config.py
=============
Tiny YAML config loader with sensible FABLE-5 defaults and RUN_MODE presets.
Everything downstream reads a plain dict, so configs stay declarative and the
run_* scripts stay thin.
"""

from __future__ import annotations

import copy
import os

import yaml

# Baseline defaults; a YAML file overrides any subset of these.
DEFAULTS = {
    "run_mode": "SMOKE",                       # SMOKE | MEDIUM | FULL
    "paths": {
        "metadata_csv": None,                  # real ISIC 2024 metadata; None -> synthetic
        "hdf5": None,                          # ISIC 2024 image HDF5; None -> synthetic/folder embeddings
        "image_dir": None,                     # optional folder of images for iToBoS/generic datasets
        "work_dir": "artifacts",               # metadata/embeddings/features/splits live here
        "results_dir": "results",
    },
    "subset": {"max_lesions": 12000, "max_patients": None},
    "data": {
        "id_col": None,                       # optional source column for lesion/image id
        "patient_col": None,                  # optional source column for patient id
        "target_col": None,                   # optional source column for malignant label
        "image_path_col": None,               # optional source column for image path/filename
        "positive_values": ["1", "true", "yes", "malignant", "melanoma", "cancer", "positive"],
    },
    "embedding": {
        "model": "resnet50",                   # resnet50 | efficientnet_b0
        "use_dino": False,                     # optional timm/DINO extractor, default OFF
        "batch_size": 64,
        "synthetic_dim": 64,                   # embedding dim used in synthetic mode
    },
    "seeds": [1, 2, 3, 4, 5, 42],
    "models": [
        "random", "classifier_only", "metadata_model", "context_only",
        "manual_fusion_validation_selected", "pointwise_logreg_fusion",
        "pairwise_rank_logreg", "pairwise_rank_mlp", "listwise_softmax_ranker",
        "lambda_pairwise_logreg",
    ],
    "pairwise": {
        "hard_negative_frac": 0.7,             # share of mined hard negatives per positive
        "max_negatives_per_positive": 40,
        "C_grid": [0.1, 1.0, 10.0],            # logreg reg, selected on validation
        "mlp_hidden": 32, "mlp_epochs": 40, "mlp_lr": 0.05, "mlp_l2": 1e-4,
    },
    "listwise": {"lr": 0.1, "epochs": 300, "l2": 1e-3},
    "eval": {
        "topk_values": [1, 3, 5, 10, 20],
        "top_pct": 0.10,                       # "recall after top 10% of a patient's lesions"
        "selection_primary": "recall@5",
        "selection_secondary": "mean_rank_first_malignant",
        "selection_tertiary": "recall@10",
        "n_bootstrap": 1000,
        "pauc_tpr_min": 0.80,                  # ISIC-2024-style partial AUC above 80% TPR
    },
    # synthetic generator (used only when metadata_csv / hdf5 are None)
    "synthetic": {"n_patients": 509, "mean_lesions": 23, "malignant_patient_frac": 0.5},
}

# RUN_MODE presets applied on top of DEFAULTS but under the YAML file.
MODE_PRESETS = {
    "SMOKE": {"subset": {"max_lesions": 12000}, "eval": {"n_bootstrap": 1000}},
    "MEDIUM": {"subset": {"max_lesions": 75000}, "eval": {"n_bootstrap": 1000}},
    "FULL": {"subset": {"max_lesions": None}, "eval": {"n_bootstrap": 1000}},
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    """DEFAULTS <- MODE_PRESETS[run_mode] <- YAML file <- explicit overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    file_cfg = {}
    if path and os.path.exists(path):
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
    run_mode = (file_cfg.get("run_mode")
                or (overrides or {}).get("run_mode") or cfg["run_mode"])
    _deep_update(cfg, MODE_PRESETS.get(run_mode, {}))
    _deep_update(cfg, file_cfg)
    _deep_update(cfg, overrides or {})
    cfg["run_mode"] = run_mode
    return cfg


# canonical artifact paths derived from a config -----------------------------
def artifact_paths(cfg):
    wd = cfg["paths"]["work_dir"]
    return {
        "work_dir": wd,
        "metadata": os.path.join(wd, "metadata.parquet"),
        "embeddings": os.path.join(wd, "embeddings.npy"),
        "emb_index": os.path.join(wd, "emb_index.json"),
        "features": os.path.join(wd, "features.parquet"),
        "splits_dir": os.path.join(wd, "splits"),
    }
