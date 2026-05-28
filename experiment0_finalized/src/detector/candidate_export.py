"""
src/detector/candidate_export.py
---------------------------------
Crops candidate boxes from source images.
Exports a candidates CSV compatible with the Stage-2 reranker.
Also exports a FP taxonomy CSV for manual labelling.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# ── Crop one bounding box from an image ──────────────────────────────────────

def crop_box(
    image_path: str,
    x1: float, y1: float, x2: float, y2: float,
    crop_size: Tuple[int, int] = (64, 64),
    pad: int = 4,
) -> Optional[Image.Image]:
    """
    Load image, crop box with padding, resize to crop_size.
    Returns None if image cannot be opened.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: cannot open {image_path}: {e}")
        return None

    w, h = img.size
    x1c = max(0, int(x1) - pad)
    y1c = max(0, int(y1) - pad)
    x2c = min(w, int(x2) + pad)
    y2c = min(h, int(y2) + pad)

    if x2c <= x1c or y2c <= y1c:
        return None

    crop = img.crop((x1c, y1c, x2c, y2c))
    crop = crop.resize(crop_size, Image.BILINEAR)
    return crop


# ── Main export ───────────────────────────────────────────────────────────────

def export_candidates(
    pred_df: pd.DataFrame,
    cfg: dict,
    threshold: float,
    split: str = "test",
) -> pd.DataFrame:
    """
    For a given threshold, crop all candidate boxes and save:
      - crops/tp/  : true positive crops
      - crops/fp/  : false positive crops
      - crops/missed/ : missed GT crops

    Returns candidate_df compatible with Stage-2 reranker CSV schema.
    """
    crops_dir   = Path(cfg["candidate_export"]["crops_dir"])
    crop_size   = tuple(cfg["candidate_export"]["crop_size"])
    cands_csv   = cfg["candidate_export"]["candidates_csv"]
    tax_csv     = cfg["candidate_export"]["fp_taxonomy_csv"]

    for sub in ["tp", "fp", "missed"]:
        (crops_dir / sub).mkdir(parents=True, exist_ok=True)

    # Filter to relevant split and threshold
    if "split" in pred_df.columns and split != "all":
        df = pred_df[pred_df["split"] == split].copy()
    else:
        df = pred_df.copy()

    actual   = df[df["score"] >= threshold].copy()
    missed   = df[df["score"] == -1.0].copy()

    candidate_rows = []
    fp_rows        = []

    # ── True positives ────────────────────────────────────────────────────────
    tp_df = actual[actual["is_tp"] == 1]
    for idx, (_, row) in enumerate(tp_df.iterrows()):
        crop = crop_box(row["image_path"],
                        row["bbox_x1"], row["bbox_y1"],
                        row["bbox_x2"], row["bbox_y2"],
                        crop_size)
        crop_path = str(crops_dir / "tp" / f"tp_{idx:05d}.png")
        if crop:
            crop.save(crop_path)
        candidate_rows.append(_make_candidate_row(
            row, crop_path, is_tp=True
        ))

    # ── False positives ───────────────────────────────────────────────────────
    fp_df = actual[actual["is_fp"] == 1]
    for idx, (_, row) in enumerate(fp_df.iterrows()):
        crop = crop_box(row["image_path"],
                        row["bbox_x1"], row["bbox_y1"],
                        row["bbox_x2"], row["bbox_y2"],
                        crop_size)
        crop_path = str(crops_dir / "fp" / f"fp_{idx:05d}.png")
        if crop:
            crop.save(crop_path)
        cand_row = _make_candidate_row(row, crop_path, is_tp=False)
        candidate_rows.append(cand_row)

        # FP taxonomy row (for manual labelling)
        fp_rows.append({
            "crop_path":     crop_path,
            "image_path":    row["image_path"],
            "image_id":      row["image_id"],
            "patient_id":    row.get("patient_id", ""),
            "score":         row["score"],
            "bbox_x1":       row["bbox_x1"],
            "bbox_y1":       row["bbox_y1"],
            "bbox_x2":       row["bbox_x2"],
            "bbox_y2":       row["bbox_y2"],
            # Manual taxonomy columns — fill these in manually
            "fp_category":   "",   # hair|shadow|skin_texture|vessel|skin_fold|
                                   # artifact|tiny_benign|border_bg|other
            "notes":         "",
        })

    # ── Missed lesions ────────────────────────────────────────────────────────
    for idx, (_, row) in enumerate(missed.iterrows()):
        crop = crop_box(row["image_path"],
                        row["bbox_x1"], row["bbox_y1"],
                        row["bbox_x2"], row["bbox_y2"],
                        crop_size)
        crop_path = str(crops_dir / "missed" / f"missed_{idx:05d}.png")
        if crop:
            crop.save(crop_path)
        candidate_rows.append(_make_candidate_row(
            row, crop_path, is_tp=False, is_missed=True
        ))

    cand_df = pd.DataFrame(candidate_rows)
    fp_df_out = pd.DataFrame(fp_rows)

    # Save
    Path(cands_csv).parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(cands_csv, index=False)
    fp_df_out.to_csv(tax_csv, index=False)

    n_tp = (cand_df["is_true_positive"] == 1).sum() if len(cand_df) > 0 else 0
    n_fp = (cand_df["is_true_positive"] == 0).sum() if len(cand_df) > 0 else 0
    n_ms = len(missed)
    print(f"\n[Candidate Export @ thresh={threshold}]")
    print(f"  TP crops  : {n_tp}  → {crops_dir}/tp/")
    print(f"  FP crops  : {n_fp}  → {crops_dir}/fp/")
    print(f"  Missed GT : {n_ms} → {crops_dir}/missed/")
    print(f"  Candidates CSV : {cands_csv}")
    print(f"  FP taxonomy    : {tax_csv}  (fill 'fp_category' column manually)")

    return cand_df


def _make_candidate_row(row, crop_path, is_tp, is_missed=False):
    """Build one row for the Stage-2-compatible candidates CSV."""
    return {
        # Stage-2 compatible schema
        "candidate_image_path": crop_path,
        "source_image_path":    row.get("image_path", ""),
        "patient_id":           row.get("patient_id", ""),
        "image_id":             row.get("image_id", ""),
        "bbox_x1":              row.get("bbox_x1", 0),
        "bbox_y1":              row.get("bbox_y1", 0),
        "bbox_x2":              row.get("bbox_x2", 0),
        "bbox_y2":              row.get("bbox_y2", 0),
        "detector_score":       float(row.get("score", -1)),
        "is_true_positive":     int(is_tp),
        "is_missed_gt":         int(is_missed),
        "matched_gt_id":        int(row.get("matched_gt_id", -1)),
        "split":                row.get("split", ""),
        # Placeholders for Stage-2 reranker columns
        "lesion_id":            f"{row.get('image_id','img')}_{row.get('bbox_x1',0):.0f}",
        "malignant":            int(is_tp),  # proxy: TP ≈ annotated lesion
        "label":                int(is_tp),
        "embedding_index":      -1,          # filled after feature extraction
        "split_reranker":       row.get("split", ""),
    }
