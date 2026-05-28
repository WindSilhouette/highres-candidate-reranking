"""
scripts/run_detector.py
-----------------------
End-to-end first-stage detector baseline runner.

Usage:
    python scripts/run_detector.py                         # toy mode
    python scripts/run_detector.py --csv path/to/data.csv  # real CSV
    python scripts/run_detector.py --mode eval_only        # skip training

Steps:
  0. Generate toy images (or load real CSV)
  1. Build datasets and dataloaders
  2. Train detector
  3. Evaluate at multiple thresholds → predictions.csv, threshold_sweep.csv
  4. Export candidate crops → candidate_crops.csv, FP taxonomy CSV
  5. Interpretability: galleries, plots, FROC, embedding projection
  6. Print bridge summary (how to pass candidates to Stage-2 reranker)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml


def load_cfg(path="configs/detector.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(
    cfg_path="configs/detector.yaml",
    csv_override=None,
    mode="full",          # "full" | "eval_only" | "export_only"
):
    cfg    = load_cfg(cfg_path)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(cfg["experiment"]["device"])
    out    = Path(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "detector_config.yaml")

    print("\n" + "█" * 70)
    print(" STAGE-1 DETECTOR BASELINE")
    print(" (diagnostic tool — motivates the second-stage reranker)")
    print("█" * 70)

    # ── Step 0: Data ──────────────────────────────────────────────────────────
    data_mode = cfg["data"]["mode"]
    if csv_override:
        data_mode = "csv"
        cfg["data"]["csv_path"] = csv_override

    if data_mode == "toy":
        print("\n[Step 0] Generating toy detection images...")
        from src.detector.toy_images import generate_toy_detection_dataset
        tc  = cfg["toy"]
        ann_df = generate_toy_detection_dataset(
            n_images=tc["n_images"],
            image_h=tc["image_h"],
            image_w=tc["image_w"],
            lesions_per_image_min=tc["lesions_per_image_min"],
            lesions_per_image_max=tc["lesions_per_image_max"],
            lesion_radius_min=tc["lesion_radius_min"],
            lesion_radius_max=tc["lesion_radius_max"],
            n_patients=tc["n_patients"],
            train_frac=tc["train_frac"],
            val_frac=tc["val_frac"],
            test_frac=tc["test_frac"],
            seed=cfg["experiment"]["seed"],
            output_dir=tc["output_dir"],
            annotations_csv=tc["annotations_csv"],
        )
        csv_path = tc["annotations_csv"]
    else:
        csv_path = cfg["data"]["csv_path"]
        print(f"\n[Step 0] Using real CSV: {csv_path}")

    # ── Step 1: Build datasets ────────────────────────────────────────────────
    print("\n[Step 1] Building datasets...")
    from src.detector.dataset import (
        CSVDetectionDataset, collate_fn, get_transform
    )
    from torch.utils.data import DataLoader

    train_ds = CSVDetectionDataset(
        csv_path, split="train",
        image_root=cfg["data"]["image_root"],
        transforms=get_transform(train=True),
    )
    val_ds = CSVDetectionDataset(
        csv_path, split="val",
        image_root=cfg["data"]["image_root"],
        transforms=get_transform(train=False),
    )
    test_ds = CSVDetectionDataset(
        csv_path, split="test",
        image_root=cfg["data"]["image_root"],
        transforms=get_transform(train=False),
    )

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} images")

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    print("\n[Step 2] Building detector model...")
    from src.detector.model import build_detector, load_checkpoint

    mc = cfg["model"]
    model = build_detector(
        model_name=mc["name"],
        num_classes=mc["num_classes"],
        pretrained_backbone=mc["pretrained_backbone"],
        nms_thresh=mc["nms_thresh"],
        score_thresh=mc["score_thresh_eval"],
        detections_per_img=mc["detections_per_img"],
        device=str(device),
    )
    print(f"  Model: {mc['name']} | device: {device}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    ckpt = cfg["training"]["checkpoint_path"]
    if mode in ("full",) and not Path(ckpt).exists():
        print("\n[Step 3] Training detector...")
        from src.detector.trainer import train_detector
        model = train_detector(model, train_loader, val_loader, cfg, device)
    elif Path(ckpt).exists():
        print(f"\n[Step 3] Loading existing checkpoint: {ckpt}")
        model, _ = load_checkpoint(model, ckpt, device)
    else:
        print("\n[Step 3] Skipping training (eval_only mode).")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    if mode in ("full", "eval_only"):
        print("\n[Step 4] Evaluating on test set...")
        from src.detector.evaluator import evaluate_detector

        ec  = cfg["evaluation"]
        pred_df, sweep_df, summary = evaluate_detector(
            model=model,
            loader=test_loader,
            thresholds=ec["thresholds"],
            iou_thresh=ec["iou_thresh_match"],
            device=device,
            operating_threshold=ec["operating_threshold"],
            predictions_path=ec["predictions_path"],
            threshold_sweep_path=ec["threshold_sweep_path"],
            metrics_path=ec["metrics_path"],
        )

    # ── Step 5: Export candidates ─────────────────────────────────────────────
    print("\n[Step 5] Exporting candidate crops...")
    from src.detector.candidate_export import export_candidates
    import pandas as pd

    if "pred_df" not in dir():
        pred_df = pd.read_csv(cfg["evaluation"]["predictions_path"])

    cand_df = export_candidates(
        pred_df=pred_df,
        cfg=cfg,
        threshold=cfg["candidate_export"]["threshold"],
        split="test",
    )

    # ── Step 6: Interpretability ──────────────────────────────────────────────
    print("\n[Step 6] Generating interpretability outputs...")
    from src.detector.interpretability import (
        plot_box_overlays, save_galleries, plot_confidence_histogram,
        plot_froc_curve, plot_embedding_projection, print_failure_summary,
    )
    ic   = cfg["interpretability"]
    plots_dir   = ic["plots_dir"]
    gallery_dir = ic["gallery_dir"]
    op_thresh   = cfg["evaluation"]["operating_threshold"]

    if "sweep_df" not in dir():
        sweep_df = pd.read_csv(cfg["evaluation"]["threshold_sweep_path"])

    plot_box_overlays(
        pred_df, threshold=op_thresh,
        output_dir=str(Path(gallery_dir) / "overlays"),
        n_images=ic["n_gallery_images"],
    )
    save_galleries(cand_df, gallery_dir=gallery_dir,
                   n=ic["n_gallery_images"])
    plot_confidence_histogram(
        pred_df, threshold=op_thresh,
        output_path=f"{plots_dir}/confidence_hist_tp_fp.png",
    )
    plot_froc_curve(
        sweep_df,
        output_dir=plots_dir,
        operating_threshold=op_thresh,
    )

    # Optional embedding projection
    _try_embedding_plot(cand_df, pred_df, model, test_loader,
                        device, cfg, ic, plots_dir)

    print_failure_summary(pred_df, threshold=op_thresh)

    # ── Step 7: Bridge summary ────────────────────────────────────────────────
    _print_bridge_summary(cfg, cand_df)

    # ── Final file list ───────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print(" STAGE-1 COMPLETE")
    print("█" * 70)
    print(f"\n  All outputs in: {out}/")
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".json", ".png", ".pt"):
            print(f"    {f.relative_to(out)}")


def _try_embedding_plot(cand_df, pred_df, model, loader,
                         device, cfg, ic, plots_dir):
    """Try to extract RoI embeddings and plot UMAP/t-SNE."""
    try:
        import numpy as np
        from src.detector.model import ROIFeatureExtractor
        from src.detector.interpretability import plot_embedding_projection

        print("  Extracting RoI features for embedding projection...")
        extractor = ROIFeatureExtractor(model, device)
        all_feats, all_labels, all_scores = [], [], []

        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images_dev = [img.to(device) for img in images]
                preds      = model(images_dev)
                for pred in preds:
                    boxes  = pred["boxes"]
                    scores = pred["scores"].cpu().numpy()
                    if len(boxes) == 0:
                        continue
                    feats = extractor.extract(images_dev, [boxes])
                    if feats is not None and len(feats) >= len(scores):
                        all_feats.append(feats[:len(scores)].numpy())
                        all_scores.extend(scores.tolist())

        extractor.remove_hooks()

        if not all_feats:
            print("  No RoI features extracted, skipping UMAP")
            return

        feats_np = np.concatenate(all_feats, axis=0)
        scores_np = np.array(all_scores)

        # Match to TP/FP labels using pred_df
        actual = pred_df[pred_df["score"] > 0].sort_values(
            "score", ascending=False
        )
        n = min(len(feats_np), len(actual))
        lab_np = np.zeros(n, dtype=int)
        tp_mask = actual["is_tp"].values[:n].astype(bool)
        lab_np[tp_mask] = 1

        method = "umap" if ic["use_umap"] else "tsne"
        plot_embedding_projection(
            feats_np[:n], lab_np, scores_np[:n],
            output_path=f"{plots_dir}/embedding_projection_{method}.png",
            method=method,
            n_neighbors=ic["umap_n_neighbors"],
        )
    except Exception as e:
        print(f"  Embedding projection skipped: {e}")


def _print_bridge_summary(cfg, cand_df):
    cands_csv = cfg["candidate_export"]["candidates_csv"]
    print("\n" + "─" * 60)
    print(" BRIDGE TO STAGE-2 RERANKER")
    print("─" * 60)
    print(f"\n  Candidate crops CSV: {cands_csv}")
    print(f"  Rows: {len(cand_df)}")
    print(f"\n  To use with the Stage-2 reranking pipeline:")
    print(f"    python run_experiment0.py --csv {cands_csv}")
    print(f"\n  The CSV contains:")
    print(f"    candidate_image_path  — path to cropped lesion image")
    print(f"    patient_id            — for patient-grouped reranking")
    print(f"    detector_score        — first-stage confidence")
    print(f"    is_true_positive      — TP/FP label (proxy: annotated = TP)")
    print(f"    lesion_id, malignant  — Stage-2 schema compatible")
    print(f"\n  ⚠️  The is_true_positive column is a DETECTION label, not a")
    print(f"     malignancy label. Replace 'malignant' with real biopsy")
    print(f"     outcomes before clinical use.")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/detector.yaml")
    parser.add_argument("--csv", default=None,
                        help="Override with real annotation CSV")
    parser.add_argument("--mode", default="full",
                        choices=["full", "eval_only", "export_only"])
    args = parser.parse_args()
    main(cfg_path=args.cfg, csv_override=args.csv, mode=args.mode)
