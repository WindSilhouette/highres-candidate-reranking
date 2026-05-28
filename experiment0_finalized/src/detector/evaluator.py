"""
src/detector/evaluator.py
-------------------------
Multi-threshold evaluation of the detector.
Primary goal: characterise the recall/FP tradeoff — not optimise mAP.

Outputs:
  - predictions.csv          (one row per predicted box)
  - threshold_sweep.csv      (recall, precision, FP/image at each threshold)
  - detector_metrics.json    (summary at operating threshold)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.detector.model import FasterRCNN, match_predictions_to_gt


# ── Full multi-threshold evaluation ──────────────────────────────────────────

@torch.no_grad()
def evaluate_detector(
    model: FasterRCNN,
    loader: DataLoader,
    thresholds: List[float],
    iou_thresh: float,
    device: torch.device,
    operating_threshold: float = 0.1,
    predictions_path: str = "detector_outputs/predictions.csv",
    threshold_sweep_path: str = "detector_outputs/threshold_sweep.csv",
    metrics_path: str = "detector_outputs/detector_metrics.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run inference at a very low score threshold, then post-filter
    to produce curves over all thresholds in one pass.

    Returns
    -------
    pred_df       : all predictions with TP/FP labels
    sweep_df      : recall, precision, FP/image per threshold
    summary       : metrics at operating_threshold
    """
    model.eval()
    min_thresh = min(thresholds + [operating_threshold]) - 0.001
    orig_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = max(0.0, min_thresh)

    all_preds = []

    print(f"\n[Detector Evaluation]")
    print(f"  IoU match threshold : {iou_thresh}")
    print(f"  Thresholds          : {thresholds}")

    n_images = 0
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        preds      = model(images_dev)

        for img, pred, tgt in zip(images, preds, targets):
            img_path  = tgt.get("image_path", "")
            patient_id = tgt.get("patient_id", "")
            split     = tgt.get("split", "")
            img_id    = tgt["image_id"].item()

            pred_boxes  = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()
            gt_boxes    = tgt["boxes"]

            # Match at lowest threshold to label each prediction
            match = match_predictions_to_gt(
                pred_boxes, pred_scores, gt_boxes, iou_thresh
            )

            for i, (box, score, label) in enumerate(
                zip(pred_boxes, pred_scores, pred_labels)
            ):
                is_tp    = i in match["tp_indices"]
                match_gt = match["matched_gt"][i]
                best_iou = match["ious"][i]
                all_preds.append({
                    "image_id":    img_id,
                    "image_path":  img_path,
                    "patient_id":  patient_id,
                    "split":       split,
                    "bbox_x1":     float(box[0]),
                    "bbox_y1":     float(box[1]),
                    "bbox_x2":     float(box[2]),
                    "bbox_y2":     float(box[3]),
                    "score":       float(score),
                    "class_id":    int(label),
                    "is_tp":       int(is_tp),
                    "is_fp":       int(not is_tp),
                    "matched_gt_id": int(match_gt),
                    "best_iou":    float(best_iou),
                    "n_gt_boxes":  len(gt_boxes),
                })

            # Track missed GTs for this image
            for gt_i in match["missed_gt"]:
                gt_box = gt_boxes[gt_i]
                all_preds.append({
                    "image_id":    img_id,
                    "image_path":  img_path,
                    "patient_id":  patient_id,
                    "split":       split,
                    "bbox_x1":     float(gt_box[0]),
                    "bbox_y1":     float(gt_box[1]),
                    "bbox_x2":     float(gt_box[2]),
                    "bbox_y2":     float(gt_box[3]),
                    "score":       -1.0,   # sentinel: missed GT
                    "class_id":    1,
                    "is_tp":       0,
                    "is_fp":       0,
                    "matched_gt_id": int(gt_i),
                    "best_iou":    0.0,
                    "n_gt_boxes":  len(gt_boxes),
                })
            n_images += 1

    model.roi_heads.score_thresh = orig_thresh

    pred_df = pd.DataFrame(all_preds)

    # Compute total number of GT boxes
    total_gt = pred_df[pred_df["score"] == -1.0]["image_id"].count() + \
               pred_df[pred_df["score"] >= 0.0].groupby("image_id").apply(
                   lambda g: int(g["n_gt_boxes"].iloc[0])
               ).sum() if len(pred_df) > 0 else 0

    # Actually recount properly
    total_gt = pred_df["n_gt_boxes"].groupby(
        pred_df["image_id"]
    ).first().sum()

    # ── Threshold sweep ───────────────────────────────────────────────────────
    actual_preds = pred_df[pred_df["score"] >= 0.0]
    sweep_rows = []
    for thresh in thresholds:
        above = actual_preds[actual_preds["score"] >= thresh]
        tp_count = above["is_tp"].sum()
        fp_count = above["is_fp"].sum()
        recall    = tp_count / max(total_gt, 1)
        precision = tp_count / max(len(above), 1)
        fp_per_img = fp_count / max(n_images, 1)
        cands_per_img = len(above) / max(n_images, 1)
        sweep_rows.append({
            "threshold":       thresh,
            "recall":          round(recall, 4),
            "precision":       round(precision, 4),
            "fp_per_image":    round(fp_per_img, 4),
            "candidates_per_image": round(cands_per_img, 4),
            "tp_count":        int(tp_count),
            "fp_count":        int(fp_count),
            "total_gt":        int(total_gt),
            "n_images":        n_images,
        })

    sweep_df = pd.DataFrame(sweep_rows)

    # ── Summary at operating threshold ────────────────────────────────────────
    op_row = sweep_df[
        sweep_df["threshold"] == operating_threshold
    ]
    if len(op_row) == 0:
        op_row = sweep_df.iloc[0:1]

    summary = {
        "operating_threshold":   operating_threshold,
        "recall":               float(op_row["recall"].iloc[0]),
        "precision":            float(op_row["precision"].iloc[0]),
        "fp_per_image":         float(op_row["fp_per_image"].iloc[0]),
        "candidates_per_image": float(op_row["candidates_per_image"].iloc[0]),
        "total_gt":             int(total_gt),
        "n_images":             n_images,
    }

    # Save
    for path, data in [
        (predictions_path, pred_df),
        (threshold_sweep_path, sweep_df),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"  Saved: {path}")

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {metrics_path}")

    print(f"\n  [Summary @ thresh={operating_threshold}]")
    print(f"    Recall          : {summary['recall']:.3f}")
    print(f"    Precision       : {summary['precision']:.3f}")
    print(f"    FP/image        : {summary['fp_per_image']:.2f}")
    print(f"    Candidates/img  : {summary['candidates_per_image']:.2f}")

    return pred_df, sweep_df, summary
