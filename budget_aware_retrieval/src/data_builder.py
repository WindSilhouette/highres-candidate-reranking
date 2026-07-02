"""
src/data_builder.py
===================
Build the FABLE-5 metadata table and patient-disjoint split files.

Real path : load ISIC 2024 / SLICE-3D metadata (Kaggle), keep one row per lesion
            with the columns we need (isic_id, patient_id, target, and any
            available context metadata), apply the RUN_MODE subset, print an
            audit, and write metadata.parquet + splits/split_seed_<s>.json.
Synthetic : if no metadata CSV is provided, generate a SLICE-3D-shaped table
            (with a learnable + context signal) so the whole pipeline runs.

Splits are written once and reused by every method/model, so all comparisons use
identical train/val/test patients.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.io_utils import write_table

# ISIC 2024 metadata columns we try to carry through if present.
META_KEEP = ["age_approx", "sex", "anatom_site_general", "tbp_lv_areaMM2",
             "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
             "tbp_lv_deltaLBnorm", "tbp_lv_norm_color", "tbp_lv_symm_2axis",
             "tbp_lv_nevi_confidence", "clin_size_long_diam_mm"]


# Common aliases across SLICE-3D / ISIC-like / iToBoS-style metadata. Config
# values in cfg["data"] override these.
ID_ALIASES = ["isic_id", "image_id", "image", "image_name", "filename", "file", "lesion_id", "id"]
PATIENT_ALIASES = ["patient_id", "patient", "subject_id", "case_id", "person_id"]
TARGET_ALIASES = ["target", "malignant", "label", "is_malignant", "melanoma", "diagnosis", "class"]
IMAGE_PATH_ALIASES = ["image_path", "path", "filepath", "file_path", "filename", "image", "image_name"]


def _first_existing(df, preferred, aliases):
    if preferred and preferred in df.columns:
        return preferred
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _target_to_binary(series, positive_values):
    # Numeric target columns are treated as >0 = malignant. String/categorical
    # targets use a conservative allow-list configurable in YAML.
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.8:
        return (num.fillna(0) > 0).astype(int)
    pos = {str(v).strip().lower() for v in (positive_values or [])}
    return series.astype(str).str.strip().str.lower().isin(pos).astype(int)


def _normalise_real_metadata(df, cfg):
    dc = cfg.get("data", {})
    id_col = _first_existing(df, dc.get("id_col"), ID_ALIASES)
    patient_col = _first_existing(df, dc.get("patient_col"), PATIENT_ALIASES)
    target_col = _first_existing(df, dc.get("target_col"), TARGET_ALIASES)
    image_path_col = _first_existing(df, dc.get("image_path_col"), IMAGE_PATH_ALIASES)

    missing = []
    if id_col is None: missing.append("id_col/isic_id/image_id/filename")
    if patient_col is None: missing.append("patient_col/patient_id/subject_id")
    if target_col is None: missing.append("target_col/target/malignant/label")
    if missing:
        raise SystemExit("[FATAL] metadata missing required field(s): " + ", ".join(missing) +
                         ". Set them under data: {id_col, patient_col, target_col, image_path_col} in the config.")

    out = pd.DataFrame({
        "isic_id": df[id_col].astype(str),
        "patient_id": df[patient_col].astype(str),
        "malignant": _target_to_binary(df[target_col], dc.get("positive_values")),
    })
    if image_path_col is not None:
        out["image_path"] = df[image_path_col].astype(str)

    # Carry through known useful metadata plus any numeric columns that are not
    # identifiers/targets. This keeps iToBoS-style handcrafted fields usable.
    for c in META_KEEP:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    excluded = {id_col, patient_col, target_col, image_path_col}
    for c in df.columns:
        if c in excluded or c in out.columns:
            continue
        if c.startswith("tbp_lv_") or pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c]
    return out


# --------------------------------------------------------------------------- #
# Synthetic ISIC-shaped metadata (used when no real CSV is given)
# --------------------------------------------------------------------------- #
def _synthetic_metadata(cfg, seed=0):
    rng = np.random.default_rng(seed)
    s = cfg["synthetic"]
    sites = ["head/neck", "upper extremity", "lower extremity",
             "torso", "palms/soles"]
    rows, lc = [], 0
    for p in range(s["n_patients"]):
        pid = f"P{p:04d}"
        n = max(3, rng.poisson(s["mean_lesions"]))
        mal_patient = rng.random() < s["malignant_patient_frac"]
        n_mal = min(n, 1 + int(rng.random() < 0.3)) if mal_patient else 0
        age = int(np.clip(rng.normal(55, 15), 15, 90))
        sex = rng.choice(["male", "female"])
        for i in range(n):
            lab = 1 if i < n_mal else 0
            rows.append({
                "isic_id": f"ISIC_{lc:07d}", "patient_id": pid, "malignant": lab,
                "age_approx": age, "sex": sex,
                "anatom_site_general": rng.choice(sites),
                "tbp_lv_areaMM2": float(abs(rng.normal(6 + 3 * lab, 3))),
                "tbp_lv_color_std_mean": float(abs(rng.normal(1.0 + 0.8 * lab, 0.6))),
                "tbp_lv_nevi_confidence": float(np.clip(rng.normal(50 - 20 * lab, 25), 0, 100)),
                "clin_size_long_diam_mm": float(abs(rng.normal(3 + 2 * lab, 1.5))),
            })
            lc += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Load + subset + audit
# --------------------------------------------------------------------------- #
def load_metadata(cfg):
    path = cfg["paths"]["metadata_csv"]
    if path and os.path.exists(path):
        print(f">> loading real metadata: {path}")
        df = pd.read_csv(path, low_memory=False)
        df = _normalise_real_metadata(df, cfg)
    else:
        print(">> no metadata_csv -> generating SYNTHETIC ISIC-shaped metadata.")
        df = _synthetic_metadata(cfg)
    df["malignant"] = df["malignant"].astype(int)
    # Stable row ids, no accidental duplicate IDs after generic filename parsing.
    if df["isic_id"].duplicated().any():
        df["isic_id"] = [f"{x}__row{i}" for i, x in enumerate(df["isic_id"].astype(str))]
    return df.reset_index(drop=True)


def apply_subset(df, cfg):
    """RUN_MODE subset. Subset by PATIENT (keeps patients whole) and always keeps
    malignant patients first so rare positives survive downsizing."""
    max_les = cfg["subset"]["max_lesions"]
    max_pat = cfg["subset"]["max_patients"]
    if not max_les and not max_pat:
        return df
    mal_patients = df.groupby("patient_id")["malignant"].max()
    order = mal_patients.sort_values(ascending=False).index.tolist()  # malignant first
    kept, n_les = [], 0
    for pid in order:
        g = df[df["patient_id"] == pid]
        if max_pat and len(kept) >= max_pat:
            break
        if max_les and n_les + len(g) > max_les and n_les > 0:
            continue
        kept.append(pid); n_les += len(g)
    return df[df["patient_id"].isin(kept)].reset_index(drop=True)


def audit(df):
    print("=" * 70); print("DATASET AUDIT"); print("=" * 70)
    per = df.groupby("patient_id").size()
    print(f"lesions                     : {len(df)}")
    print(f"patients                    : {df['patient_id'].nunique()}")
    print(f"malignant lesions           : {int(df['malignant'].sum())} "
          f"({100*df['malignant'].mean():.2f}%)")
    print(f"malignant patients          : {int(df.groupby('patient_id')['malignant'].max().sum())}")
    print(f"lesions/patient (min/med/max): {per.min()} / {per.median():.0f} / {per.max()}")
    print("columns                     :", list(df.columns))
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        print("metadata missingness (top)  :")
        for c, v in miss.head(8).items():
            print(f"    {c:34s} {100*v:5.1f}%")
    else:
        print("metadata missingness        : none")
    print()


# --------------------------------------------------------------------------- #
# Patient-disjoint splits (stratified on malignant-patient flag), per seed
# --------------------------------------------------------------------------- #
def make_split(df, seed, val_frac=0.2, test_frac=0.2):
    pats = df.groupby("patient_id")["malignant"].max().reset_index()
    # coerce to plain numpy (pandas may use pyarrow-backed strings, which break
    # sklearn's array indexing)
    ids = np.asarray(pats["patient_id"].tolist(), dtype=object)
    strat = np.asarray(pats["malignant"].astype(int).tolist(), dtype=int)
    s = strat if (strat.sum() >= 2 and (strat == 0).sum() >= 2) else None
    p_tv, p_te = train_test_split(ids, test_size=test_frac, random_state=seed, stratify=s)
    s_tv = (np.asarray(pats.set_index("patient_id").loc[p_tv, "malignant"].astype(int).tolist(), dtype=int)
            if s is not None else None)
    if s_tv is not None and (s_tv.sum() < 2 or (s_tv == 0).sum() < 2):
        s_tv = None
    p_tr, p_va = train_test_split(p_tv, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=s_tv)
    tr, va, te = set(p_tr), set(p_va), set(p_te)
    assert not (tr & va) and not (tr & te) and not (va & te), "PATIENT LEAKAGE"
    return {"seed": seed, "train": sorted(tr), "val": sorted(va), "test": sorted(te)}


def build(cfg):
    from src.config import artifact_paths
    ap = artifact_paths(cfg)
    os.makedirs(ap["work_dir"], exist_ok=True)
    os.makedirs(ap["splits_dir"], exist_ok=True)

    df = apply_subset(load_metadata(cfg), cfg)
    audit(df)
    write_table(df, ap["metadata"], index=False)
    print(f">> wrote {ap['metadata']} ({len(df)} lesions)")

    for seed in cfg["seeds"]:
        sp = make_split(df, seed)
        with open(os.path.join(ap["splits_dir"], f"split_seed_{seed}.json"), "w") as f:
            json.dump(sp, f)
    print(f">> wrote {len(cfg['seeds'])} split files -> {ap['splits_dir']}/\n")
    return df
