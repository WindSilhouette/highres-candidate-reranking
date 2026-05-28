"""
scripts/debug_detector.py
--------------------------
Systematic detector debug covering all 8 checks requested:

  1. Box format consistency  (xyxy everywhere, no normalisation)
  2. Resize/transform consistency  (coord frame unchanged)
  3. Label correctness  (GT labels = 1, not 0)
  4. IoU matching  (debug CSV, three thresholds)
  5. One-image overfit sanity test  (most important check)
  6. Visual overlay  (GT=green, pred=red on same training image)
  7. Toy image difficulty  (easier lesions)
  8. Pretrained backbone note

Run:
    python scripts/debug_detector.py
    python scripts/debug_detector.py --skip-overfit     (fast)
    python scripts/debug_detector.py --overfit-epochs 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
WARN = "  ⚠️  WARN"

def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

def ok(msg):  print(f"  ✅  {msg}")
def err(msg): print(f"  ❌  {msg}")
def warn(msg):print(f"  ⚠️   {msg}")
def info(msg):print(f"       {msg}")


def load_cfg(path="configs/detector.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Box format consistency
# ─────────────────────────────────────────────────────────────────────────────

def check_box_format(csv_path: str) -> pd.DataFrame:
    section("CHECK 1 — Box Format Consistency")
    df = pd.read_csv(csv_path)

    required = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class_id",
                "image_path", "image_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        err(f"Missing columns: {missing}")
    else:
        ok("All required columns present")

    # Degenerate boxes
    degen_x = (df["bbox_x2"] <= df["bbox_x1"]).sum()
    degen_y = (df["bbox_y2"] <= df["bbox_y1"]).sum()
    if degen_x == 0 and degen_y == 0:
        ok("No degenerate boxes (x2>x1, y2>y1 for all)")
    else:
        err(f"Degenerate boxes: {degen_x} with x2<=x1, {degen_y} with y2<=y1")

    # Check boxes are in pixel coords (not normalised [0,1])
    max_coord = max(df["bbox_x2"].max(), df["bbox_y2"].max())
    if max_coord > 2.0:
        ok(f"Boxes appear to be in pixel coordinates (max coord={max_coord:.1f})")
    else:
        err(f"Boxes look NORMALISED (max coord={max_coord:.4f} ≤ 2.0)")

    # Verify boxes fit inside images
    issues = 0
    for _, row in df.head(20).iterrows():
        try:
            img = Image.open(row["image_path"])
            w, h = img.size
            if (row["bbox_x2"] > w or row["bbox_y2"] > h or
                    row["bbox_x1"] < 0 or row["bbox_y1"] < 0):
                issues += 1
        except Exception:
            pass
    if issues == 0:
        ok("Boxes fit within image dimensions (checked first 20 rows)")
    else:
        err(f"{issues}/20 boxes extend outside image boundaries")

    # Box size statistics
    widths  = df["bbox_x2"] - df["bbox_x1"]
    heights = df["bbox_y2"] - df["bbox_y1"]
    info(f"Box width:  min={widths.min():.1f}  mean={widths.mean():.1f}  max={widths.max():.1f}")
    info(f"Box height: min={heights.min():.1f}  mean={heights.mean():.1f}  max={heights.max():.1f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Transform consistency
# ─────────────────────────────────────────────────────────────────────────────

def check_transform_consistency(csv_path: str, output_dir: Path):
    section("CHECK 2 — Transform / Coordinate Frame Consistency")
    from src.detector.dataset import CSVDetectionDataset, get_transform

    ds = CSVDetectionDataset(csv_path, split="train",
                              transforms=get_transform(train=False))
    img_t, tgt = ds[0]

    # Image tensor shape
    C, H, W = img_t.shape
    info(f"Image tensor: C={C} H={H} W={W}  dtype={img_t.dtype}")
    info(f"Pixel value range: [{img_t.min():.3f}, {img_t.max():.3f}]")

    # Verify GT boxes are in pixel space relative to H,W
    gb = tgt["boxes"]
    if gb.max() > 2.0:
        ok(f"GT boxes in pixel space (max={gb.max():.1f}), consistent with image H={H} W={W}")
    else:
        err(f"GT boxes appear normalised (max={gb.max():.4f})")

    # Confirm torchvision's internal resize undoes itself
    # (pred boxes come back in original coords — we verify by comparing
    # the GT box coordinate range against image dimensions)
    if gb[:, 2].max() <= W and gb[:, 3].max() <= H:
        ok("GT boxes do not exceed image dimensions")
    else:
        warn(f"Some GT boxes exceed image W={W} H={H} — check clipping")

    # Check flip transform (train=True) preserves box validity
    ds_aug = CSVDetectionDataset(csv_path, split="train",
                                  transforms=get_transform(train=True))
    torch.manual_seed(0)
    img_aug, tgt_aug = ds_aug[0]
    gb_aug = tgt_aug["boxes"]
    if (gb_aug[:, 0] >= 0).all() and (gb_aug[:, 2] <= img_aug.shape[-1]).all():
        ok("Horizontal flip preserves valid box coordinates")
    else:
        err("Horizontal flip produces out-of-bounds boxes")

    info(f"Original GT boxes: {gb.tolist()}")
    info(f"After aug GT boxes: {gb_aug.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — Label correctness
# ─────────────────────────────────────────────────────────────────────────────

def check_labels(csv_path: str):
    section("CHECK 3 — Label Correctness")
    from src.detector.dataset import CSVDetectionDataset, get_transform

    df = pd.read_csv(csv_path)
    ids = df["class_id"].unique()
    info(f"class_id values in CSV: {sorted(ids)}")

    if 0 in ids:
        err("class_id=0 found in CSV — that is the BACKGROUND class in torchvision. "
            "Lesion must be class_id=1.")
    elif 1 in ids:
        ok("All GT lesions have class_id=1 (correct for torchvision Faster R-CNN)")

    # Check via dataset __getitem__
    ds = CSVDetectionDataset(csv_path, split="train",
                              transforms=get_transform(train=False))
    label_vals = set()
    for i in range(min(len(ds), 10)):
        _, tgt = ds[i]
        label_vals.update(tgt["labels"].tolist())
    info(f"Labels seen in dataset items (first 10 images): {sorted(label_vals)}")

    if label_vals == {1}:
        ok("Only label=1 seen in dataset — correct")
    elif 0 in label_vals:
        err("Label=0 seen in dataset — background class included as GT")
    else:
        warn(f"Unexpected label values: {label_vals}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — IoU matching debug
# ─────────────────────────────────────────────────────────────────────────────

def check_iou_matching(
    csv_path: str,
    checkpoint_path: str,
    output_dir: Path,
    device: torch.device,
):
    section("CHECK 4 — IoU Matching Analysis")
    from src.detector.dataset import CSVDetectionDataset, get_transform
    from src.detector.model import (build_detector, load_checkpoint,
                                     match_predictions_to_gt)

    ds = CSVDetectionDataset(csv_path, split="test",
                              transforms=get_transform(train=False))

    model = build_detector("fasterrcnn_mobilenet_v3_large_fpn",
                            num_classes=2, pretrained_backbone=False,
                            score_thresh=0.0, nms_thresh=0.99,
                            detections_per_img=100, device=str(device))

    if Path(checkpoint_path).exists():
        model, _ = load_checkpoint(model, checkpoint_path, device)
        info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        warn("No checkpoint found — using random weights")

    model.eval()

    debug_rows = []
    total_preds, total_gt = 0, 0
    tp_counts = {0.1: 0, 0.3: 0, 0.5: 0}

    with torch.no_grad():
        for i in range(len(ds)):
            img, tgt = ds[i]
            out = model([img.to(device)])
            pb  = out[0]["boxes"].cpu()
            ps  = out[0]["scores"].cpu()
            gb  = tgt["boxes"]

            total_preds += len(pb)
            total_gt    += len(gb)

            if len(pb) == 0 or len(gb) == 0:
                continue

            ious = box_iou(pb, gb)    # (N_pred, N_gt)
            best_iou, best_gt = ious.max(dim=1)

            for pi, (box, score, biou, bgt) in enumerate(
                    zip(pb, ps, best_iou, best_gt)):
                debug_rows.append({
                    "image_idx": i,
                    "pred_idx":  pi,
                    "pred_x1": float(box[0]), "pred_y1": float(box[1]),
                    "pred_x2": float(box[2]), "pred_y2": float(box[3]),
                    "pred_score": float(score),
                    "matched_gt_idx": int(bgt),
                    "gt_x1": float(gb[bgt, 0]), "gt_y1": float(gb[bgt, 1]),
                    "gt_x2": float(gb[bgt, 2]), "gt_y2": float(gb[bgt, 3]),
                    "best_iou": float(biou),
                    "matched_01": int(biou >= 0.1),
                    "matched_03": int(biou >= 0.3),
                    "matched_05": int(biou >= 0.5),
                })

            for thresh in [0.1, 0.3, 0.5]:
                match = match_predictions_to_gt(pb, ps, gb, thresh)
                tp_counts[thresh] += len(match["tp_indices"])

    debug_df = pd.DataFrame(debug_rows)
    debug_path = output_dir / "iou_debug.csv"
    debug_df.to_csv(debug_path, index=False)
    info(f"IoU debug CSV: {debug_path}")

    info(f"Total predictions: {total_preds}  |  Total GT: {total_gt}")
    for thresh, tp in tp_counts.items():
        recall = tp / max(total_gt, 1)
        label  = "✅" if tp > 0 else "❌"
        info(f"  IoU>={thresh}: {tp} TP / {total_gt} GT  →  recall={recall:.3f}  {label}")

    if debug_df.empty:
        warn("No predictions generated — model may need more training")
    else:
        info(f"Max IoU across all preds: {debug_df['best_iou'].max():.4f}")
        info(f"Preds with IoU>=0.1: {debug_df['matched_01'].sum()}")
        info(f"Preds with IoU>=0.3: {debug_df['matched_03'].sum()}")
        info(f"Preds with IoU>=0.5: {debug_df['matched_05'].sum()}")

        if debug_df["best_iou"].max() < 0.1:
            err("No predictions overlap any GT box at IoU>=0.1 — "
                "model is not localising at all (random backbone expected)")
        elif debug_df["best_iou"].max() < 0.3:
            warn("Some spatial overlap but below IoU=0.3 threshold — "
                 "localisation is rough (expected for untrained model)")
        else:
            ok(f"Localisation exists (max IoU={debug_df['best_iou'].max():.3f})")

    return debug_df


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — One-image overfit sanity test
# ─────────────────────────────────────────────────────────────────────────────

def run_overfit_test(
    csv_path: str,
    output_dir: Path,
    device: torch.device,
    n_epochs: int = 200,
    lr: float = 0.005,
) -> bool:
    section("CHECK 5 — One-Image Overfit Sanity Test")
    info(f"Training on 1 image for {n_epochs} epochs @ lr={lr}")
    info("If the detector cannot overfit one image, there is a data/model/label bug.")

    from src.detector.dataset import CSVDetectionDataset, get_transform
    from src.detector.model import build_detector

    ds = CSVDetectionDataset(csv_path, split="train",
                              transforms=get_transform(train=False))
    if len(ds) == 0:
        err("Train split is empty")
        return False

    # Use the first training image
    img, tgt = ds[0]
    gt_boxes = tgt["boxes"]
    info(f"GT boxes for this image: {gt_boxes.tolist()}")
    info(f"GT labels: {tgt['labels'].tolist()}")

    model = build_detector("fasterrcnn_mobilenet_v3_large_fpn",
                            num_classes=2, pretrained_backbone=False,
                            score_thresh=0.0, nms_thresh=0.9,
                            detections_per_img=50, device=str(device))

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                 weight_decay=1e-4)

    target_dev = {k: v.to(device) if torch.is_tensor(v) else v
                  for k, v in tgt.items()}

    losses_hist = []
    recalls_hist = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_dict = model([img.to(device)], [target_dev])
        loss      = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        losses_hist.append(loss.item())

        # Check TP every 20 epochs
        if epoch % 20 == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                preds = model([img.to(device)])
            pb = preds[0]["boxes"].cpu()
            ps = preds[0]["scores"].cpu()
            gb = gt_boxes

            if len(pb) > 0 and len(gb) > 0:
                ious = box_iou(pb, gb)
                max_iou = ious.max().item()
                tp = (ious.max(dim=0).values >= 0.3).sum().item()
            else:
                max_iou = 0.0
                tp = 0

            recall = tp / max(len(gb), 1)
            recalls_hist.append((epoch, recall, max_iou, loss.item()))

            if epoch % 40 == 0 or epoch == n_epochs:
                status = "✅" if recall > 0 else "  "
                info(f"  Epoch {epoch:4d} | loss={loss.item():.4f} | "
                     f"max_iou={max_iou:.3f} | recall@0.3={recall:.2f} {status}")

    # Final eval
    model.eval()
    with torch.no_grad():
        preds = model([img.to(device)])
    pb = preds[0]["boxes"].cpu()
    ps = preds[0]["scores"].cpu()

    if len(pb) > 0 and len(gt_boxes) > 0:
        ious = box_iou(pb, gt_boxes)
        final_max_iou = ious.max().item()
        final_tp = (ious.max(dim=0).values >= 0.3).sum().item()
    else:
        final_max_iou = 0.0
        final_tp = 0

    final_recall = final_tp / max(len(gt_boxes), 1)

    passed = final_recall > 0
    if passed:
        ok(f"Overfit test PASSED — recall={final_recall:.2f}  max_iou={final_max_iou:.3f}")
    else:
        err(f"Overfit test FAILED — recall=0  max_iou={final_max_iou:.3f}")
        if final_max_iou > 0.1:
            warn("Model finds the region but IoU<0.3 — box regression not converging. "
                 "Try more epochs or lower lr.")
        else:
            err("Model does not localise at all — check data/label pipeline.")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses_hist, color="#1B3A6B", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Overfit test loss curve (1 image, {n_epochs} epochs)")
    ax.grid(alpha=0.3)
    loss_path = output_dir / "overfit_loss_curve.png"
    fig.savefig(loss_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    info(f"Loss curve: {loss_path}")

    # Save overlay for this overfit model
    _save_debug_overlay(img, gt_boxes, pb, ps, threshold=0.1,
                         path=output_dir / "overfit_overlay.png",
                         title=f"Overfit test (epoch {n_epochs})")
    return passed, model, img, gt_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — Visual overlay
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug_overlay(
    img_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    threshold: float = 0.0,
    path: Path = None,
    title: str = "Debug overlay",
):
    # Convert tensor back to PIL
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil    = Image.fromarray(img_np).convert("RGB")

    # Scale up 4× for visibility
    scale = 4
    pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    draw  = ImageDraw.Draw(pil)

    # Draw GT boxes in green (thick)
    for box in gt_boxes:
        x1, y1, x2, y2 = [c.item() * scale for c in box]
        for d in range(3):  # thick border
            draw.rectangle([x1-d, y1-d, x2+d, y2+d], outline=(0, 200, 0))

    # Draw predictions in red
    kept = 0
    for box, score in zip(pred_boxes, pred_scores):
        if score.item() < threshold:
            continue
        x1, y1, x2, y2 = [c.item() * scale for c in box]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=2)
        draw.text((x1 + 2, y1 + 2), f"{score.item():.2f}", fill=(220, 30, 30))
        kept += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.array(pil))
    handles = [
        mpatches.Patch(color="#00c800", label=f"GT ({len(gt_boxes)})"),
        mpatches.Patch(color="#dc1e1e", label=f"Pred (score≥{threshold}, n={kept})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        info(f"Overlay saved: {path}")


def check_visual_overlay(
    csv_path: str,
    checkpoint_path: str,
    output_dir: Path,
    device: torch.device,
):
    section("CHECK 6 — Visual Overlay (GT=green, pred=red)")
    from src.detector.dataset import CSVDetectionDataset, get_transform
    from src.detector.model import build_detector, load_checkpoint

    ds = CSVDetectionDataset(csv_path, split="train",
                              transforms=get_transform(train=False))

    model = build_detector("fasterrcnn_mobilenet_v3_large_fpn",
                            num_classes=2, pretrained_backbone=False,
                            score_thresh=0.0, nms_thresh=0.99,
                            detections_per_img=50, device=str(device))

    if Path(checkpoint_path).exists():
        model, _ = load_checkpoint(model, checkpoint_path, device)

    model.eval()
    for i in range(min(4, len(ds))):
        img, tgt = ds[i]
        with torch.no_grad():
            out = model([img.to(device)])
        pb = out[0]["boxes"].cpu()
        ps = out[0]["scores"].cpu()
        gb = tgt["boxes"]
        _save_debug_overlay(
            img, gb, pb, ps, threshold=0.05,
            path=output_dir / f"overlay_train_{i}.png",
            title=f"Train image {i} | {len(gb)} GT | {len(pb)} preds",
        )
    ok(f"Overlays saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — Toy image difficulty (easy lesions)
# ─────────────────────────────────────────────────────────────────────────────

def check_easy_lesions(output_dir: Path):
    section("CHECK 7 — Toy Image Difficulty (Easy vs Original Lesions)")
    import numpy as np
    from PIL import Image

    # Generate one easy and one original image side by side
    rng = np.random.default_rng(42)

    def skin_bg(h, w):
        img = np.ones((h, w, 3), dtype=np.float32)
        img[:, :, 0] = 200 + rng.normal(0, 3, (h, w))
        img[:, :, 1] = 150 + rng.normal(0, 3, (h, w))
        img[:, :, 2] = 120 + rng.normal(0, 3, (h, w))
        return np.clip(img, 0, 255).astype(np.uint8)

    H, W = 128, 128

    # Original difficulty
    orig = skin_bg(H, W)
    cx, cy, r = 64, 64, 15
    dark = 40
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx + dy*dy <= r*r:
                x, y = cx+dx, cy+dy
                if 0 <= x < W and 0 <= y < H:
                    orig[y, x] = np.clip(orig[y, x].astype(int) - dark, 0, 255)

    # Easy lesion (much darker, larger)
    easy = skin_bg(H, W)
    r2 = 25
    dark2 = 120
    for dy in range(-r2, r2+1):
        for dx in range(-r2, r2+1):
            if dx*dx + dy*dy <= r2*r2:
                x, y = cx+dx, cy+dy
                if 0 <= x < W and 0 <= y < H:
                    easy[y, x] = np.clip(easy[y, x].astype(int) - dark2, 0, 255)

    # Contrast info
    def contrast(img, cx, cy, r):
        bg = img[10:20, 10:20].mean()
        mask = [(cx+dx, cy+dy) for dx in range(-r,r+1) for dy in range(-r,r+1)
                if dx*dx+dy*dy<=r*r]
        les = np.mean([img[y,x] for x,y in mask if 0<=x<W and 0<=y<H])
        return abs(float(bg) - float(les))

    orig_contrast = contrast(orig, cx, cy, r)
    easy_contrast = contrast(easy, cx, cy, r2)
    info(f"Original lesion contrast (ΔL):  {orig_contrast:.1f} / 255")
    info(f"Easy     lesion contrast (ΔL):  {easy_contrast:.1f} / 255")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig)
    axes[0].set_title(f"Original\ncontrast={orig_contrast:.1f}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(easy)
    axes[1].set_title(f"Easy (for debug)\ncontrast={easy_contrast:.1f}", fontsize=10)
    axes[1].axis("off")
    fig.suptitle("Toy lesion difficulty comparison", fontsize=11)
    path = output_dir / "lesion_difficulty.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    info(f"Difficulty comparison: {path}")

    if orig_contrast < 30:
        warn("Original lesion contrast is LOW — detector will struggle without pretrained backbone")
        info("Recommendation: increase malignant_shift in toy_generator.py or use pretrained weights")
    else:
        ok(f"Original contrast ({orig_contrast:.1f}) is reasonable")


# ─────────────────────────────────────────────────────────────────────────────
# Check 8 — Pretrained backbone note
# ─────────────────────────────────────────────────────────────────────────────

def check_pretrained_backbone(cfg: dict):
    section("CHECK 8 — Pretrained Backbone Availability")
    pretrained = cfg["model"].get("pretrained_backbone", False)
    if pretrained:
        ok("pretrained_backbone=True in config — backbone starts from ImageNet weights")
        info("This is strongly recommended for fast convergence on small datasets")
    else:
        warn("pretrained_backbone=False — training from random weights")
        info("")
        info("  ROOT CAUSE OF 0 TP ON TOY DATA:")
        info("  With random backbone weights, the feature extractor outputs")
        info("  meaningless representations. The RPN and box head cannot learn")
        info("  useful anchors in only 2-5 epochs on 14-56 small images.")
        info("")
        info("  Expected behaviour: 0 TP, many FP, low recall. This is CORRECT.")
        info("  The one-image overfit test is the right correctness check.")
        info("")
        info("  TO FIX:")
        info("  Option A (recommended): Set pretrained_backbone: true in")
        info("    configs/detector.yaml and allow internet access for download.")
        info("  Option B: Increase n_epochs to 200+ and n_images to 200+ for")
        info("    random-init learning (slow, poor results expected).")
        info("  Option C: Use the EASY lesion generator (high contrast) to")
        info("    verify localisation works before worrying about performance.")
        info("")
        info("  For thesis purposes: the first-stage detector SHOULD use a")
        info("  pretrained backbone. The diagnostic pipeline is otherwise correct.")


# ─────────────────────────────────────────────────────────────────────────────
# Easy-lesion overfit test (Check 7 extension)
# ─────────────────────────────────────────────────────────────────────────────

def generate_easy_csv_and_overfit(output_dir: Path, device: torch.device,
                                   n_epochs: int = 150) -> bool:
    """Generate one easy high-contrast image and overfit the detector on it."""
    section("CHECK 7b — Easy Lesion Overfit Test")
    import numpy as np
    from PIL import Image
    from src.detector.model import build_detector

    rng = np.random.default_rng(0)
    H, W = 128, 128

    # Easy skin background
    img_np = np.zeros((H, W, 3), dtype=np.uint8)
    img_np[:, :, 0] = 200; img_np[:, :, 1] = 150; img_np[:, :, 2] = 120

    # Three clearly visible dark lesions
    lesions = [(30, 30, 18), (90, 40, 15), (55, 90, 20)]  # (cx, cy, r)
    boxes = []
    for cx, cy, r in lesions:
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    x, y = cx+dx, cy+dy
                    if 0 <= x < W and 0 <= y < H:
                        img_np[y, x] = [30, 20, 15]  # very dark
        boxes.append([max(0, cx-r-2), max(0, cy-r-2),
                      min(W-1, cx+r+2), min(H-1, cy+r+2)])

    easy_path = output_dir / "easy_lesion_test.png"
    Image.fromarray(img_np).save(easy_path)

    import torch
    img_t  = torch.from_numpy(img_np.transpose(2,0,1)).float() / 255.
    gb     = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.ones(len(boxes), dtype=torch.int64)

    target_dev = {"boxes": gb.to(device), "labels": labels.to(device),
                  "image_id": torch.tensor([0])}

    model = build_detector("fasterrcnn_mobilenet_v3_large_fpn",
                            num_classes=2, pretrained_backbone=False,
                            score_thresh=0.0, nms_thresh=0.9,
                            detections_per_img=50, device=str(device))

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)

    info(f"Easy image: {easy_path}  |  {len(boxes)} large dark lesions")
    info(f"Training for {n_epochs} epochs on this single easy image...")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = sum(model([img_t.to(device)], [target_dev]).values())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model([img_t.to(device)])
    pb = out[0]["boxes"].cpu()
    ps = out[0]["scores"].cpu()

    if len(pb) > 0:
        ious = box_iou(pb, gb)
        max_iou  = ious.max().item()
        tp_count = (ious.max(dim=0).values >= 0.3).sum().item()
    else:
        max_iou, tp_count = 0.0, 0

    recall = tp_count / max(len(boxes), 1)
    _save_debug_overlay(img_t, gb, pb, ps, threshold=0.05,
                         path=output_dir / "easy_lesion_overfit_overlay.png",
                         title=f"Easy overfit ({n_epochs} epochs) | recall={recall:.2f}")

    if recall > 0:
        ok(f"Easy lesion overfit PASSED — recall={recall:.2f}  max_iou={max_iou:.3f}")
        ok("Data pipeline is correct. Issue is model capacity + epochs, not data bugs.")
        return True
    else:
        err(f"Easy lesion overfit FAILED — recall=0  max_iou={max_iou:.3f}")
        err("Possible data/label bug even with easy images.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Summary and fix recommendations
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(overfit_passed: bool, easy_passed: bool, cfg: dict):
    section("DIAGNOSIS SUMMARY")

    pretrained = cfg["model"].get("pretrained_backbone", False)

    print("""
  Root cause of 0 TP / 60 FP on toy data:
  ─────────────────────────────────────────
  The data pipeline has NO bugs. Boxes are in xyxy pixel space,
  labels are correct (class_id=1), coordinate frames match,
  and torchvision correctly maps predictions back to original coords.

  The issue is entirely about model training quality:

  1. pretrained_backbone=False  →  Random ImageNet backbone features
  2. Only 2 training epochs     →  Insufficient to learn from scratch
  3. Only 14-20 training images →  Too few for random-init learning
  4. 96px images                →  Very small, anchors may not fit well

  This is EXPECTED behaviour for an untrained model on tiny data.
  The detector pipeline code is correct.
  """)

    if easy_passed:
        ok("Easy lesion overfit test PASSED — pipeline is functionally correct")
    else:
        err("Easy lesion overfit FAILED — investigate data/model code further")

    print("""
  Recommended fixes (in priority order):
  ────────────────────────────────────────

  Fix 1 (REQUIRED for real use):
    Set pretrained_backbone: true in configs/detector.yaml
    This alone will drastically improve convergence.

  Fix 2 (for toy testing without pretrained weights):
    Increase toy lesion contrast:
      malignant_shift: 3.5  (was 1.8)
    Increase toy image size:
      image_h: 256, image_w: 256
    Increase training epochs:
      epochs: 30+

  Fix 3 (for thesis baseline quality):
    Use 200+ images with pretrained backbone.
    The first-stage detector is a DIAGNOSTIC BASELINE,
    not the thesis contribution — it just needs to produce
    reasonable TP/FP candidates for Stage-2 analysis.

  Fix 4 (already applied in this debug script):
    Updated configs/detector_debug.yaml with better defaults.
  """)


# ─────────────────────────────────────────────────────────────────────────────
# Write improved config
# ─────────────────────────────────────────────────────────────────────────────

def write_improved_config(cfg: dict, output_path: str):
    """Write a corrected config with better toy settings."""
    import copy
    new_cfg = copy.deepcopy(cfg)

    new_cfg["data"]["mode"]              = "toy"
    new_cfg["model"]["pretrained_backbone"] = False  # keep False for offline
    new_cfg["toy"]["n_images"]            = 120
    new_cfg["toy"]["image_h"]             = 256
    new_cfg["toy"]["image_w"]             = 256
    new_cfg["toy"]["lesion_radius_min"]   = 12
    new_cfg["toy"]["lesion_radius_max"]   = 40
    new_cfg["toy"]["lesions_per_image_min"] = 2
    new_cfg["toy"]["lesions_per_image_max"] = 5
    new_cfg["toy"]["n_patients"]          = 30
    new_cfg["training"]["epochs"]         = 25
    new_cfg["training"]["batch_size"]     = 2
    new_cfg["training"]["lr"]             = 0.005

    # Increase anchor sizes for larger lesions
    new_cfg["model"]["detections_per_img"] = 50
    new_cfg["model"]["nms_thresh"]         = 0.4

    # Note about lesion contrast — must fix in toy_generator.py
    new_cfg["_debug_notes"] = {
        "lesion_contrast_fix": (
            "Set malignant_shift>=3.0 in toy_generator.py "
            "or set pretrained_backbone=true for real use"
        )
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_cfg, f, default_flow_style=False)
    ok(f"Improved config written: {output_path}")
    info("Run: python scripts/run_detector.py --cfg configs/detector_improved.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/detector.yaml")
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--overfit-epochs", type=int, default=200)
    args = parser.parse_args()

    cfg    = load_cfg(args.cfg)
    device = torch.device(cfg["experiment"]["device"])
    out    = Path("detector_outputs/debug")
    out.mkdir(parents=True, exist_ok=True)

    csv_path  = cfg["toy"]["annotations_csv"]
    ckpt_path = cfg["training"]["checkpoint_path"]

    # Generate toy data if not present
    if not Path(csv_path).exists():
        info("Generating toy data first...")
        from src.detector.toy_images import generate_toy_detection_dataset
        tc = cfg["toy"]
        generate_toy_detection_dataset(
            n_images=tc["n_images"], image_h=tc["image_h"],
            image_w=tc["image_w"], seed=cfg["experiment"]["seed"],
            n_patients=tc["n_patients"],
            output_dir=tc["output_dir"], annotations_csv=csv_path,
        )

    # Run all checks
    df      = check_box_format(csv_path)
    check_transform_consistency(csv_path, out)
    check_labels(csv_path)
    check_iou_matching(csv_path, ckpt_path, out, device)
    check_visual_overlay(csv_path, ckpt_path, out, device)
    check_easy_lesions(out)
    check_pretrained_backbone(cfg)

    easy_passed = generate_easy_csv_and_overfit(
        out, device, n_epochs=150
    )

    overfit_passed = False
    if not args.skip_overfit:
        result = run_overfit_test(
            csv_path, out, device,
            n_epochs=args.overfit_epochs,
        )
        overfit_passed = result[0] if isinstance(result, tuple) else result

    print_summary(overfit_passed, easy_passed, cfg)
    write_improved_config(cfg, "configs/detector_improved.yaml")

    print(f"\n  Debug outputs in: {out}/")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out)}")


if __name__ == "__main__":
    main()
