"""
src/detector/trainer.py
-----------------------
Training loop for Faster R-CNN detector.
Prioritises recall over mAP — uses low score thresholds and
evaluates on recall@0.3IoU as the primary stopping metric.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.detector.dataset import collate_fn
from src.detector.model import FasterRCNN, save_checkpoint, match_predictions_to_gt


# ── Training loop ─────────────────────────────────────────────────────────────

def train_detector(
    model: FasterRCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> FasterRCNN:
    """
    Train the detector. Saves best checkpoint by val recall.

    Returns the model with best weights loaded.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=cfg["training"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"],
    )

    best_recall  = -1.0
    ckpt_path    = cfg["training"]["checkpoint_path"]
    epochs       = cfg["training"]["epochs"]
    eval_every   = cfg["training"]["eval_every"]
    grad_clip    = cfg["training"]["grad_clip"]
    iou_thresh   = cfg["evaluation"]["iou_thresh_match"]

    print("\n[Detector Training]")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {cfg['training']['lr']}")

    history = []

    for epoch in range(1, epochs + 1):
        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            # Strip non-tensor keys — torchvision only needs 'boxes' and 'labels'
            targets = [
                {k: v.to(device)
                 for k, v in t.items()
                 if torch.is_tensor(v) and k in ("boxes", "labels", "image_id")}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            if torch.isnan(losses):
                print(f"  Warning: NaN loss at epoch {epoch}, skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            epoch_loss += losses.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_recall = None
        if epoch % eval_every == 0 or epoch == epochs:
            val_recall = _eval_recall(
                model, val_loader, device, iou_thresh,
                score_thresh=cfg["evaluation"]["operating_threshold"],
            )
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_recall": val_recall,
            })
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                  f"val_recall={val_recall:.3f}")

            if val_recall > best_recall:
                best_recall = val_recall
                save_checkpoint(model, ckpt_path, epoch, avg_loss)
        else:
            history.append({"epoch": epoch, "train_loss": avg_loss})
            if epoch % max(1, eval_every // 2) == 0:
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

    print(f"\n  Best val recall: {best_recall:.3f}")

    # Save history
    hist_path = Path(cfg["data"]["output_dir"]) / "training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2))

    # Load best
    from src.detector.model import load_checkpoint
    model, _ = load_checkpoint(model, ckpt_path, device)
    return model


# ── Recall evaluation helper ──────────────────────────────────────────────────

@torch.no_grad()
def _eval_recall(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    iou_thresh: float,
    score_thresh: float = 0.1,
) -> float:
    """Compute lesion-level recall on validation set."""
    model.eval()
    total_gt = 0
    total_tp = 0

    # Temporarily lower score threshold for eval
    orig = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = score_thresh

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, tgt in zip(preds, targets):
            gt_boxes   = tgt["boxes"]
            pred_boxes = pred["boxes"].cpu()
            pred_scores= pred["scores"].cpu()

            total_gt += len(gt_boxes)
            if len(pred_boxes) == 0:
                continue

            match = match_predictions_to_gt(
                pred_boxes, pred_scores, gt_boxes, iou_thresh
            )
            total_tp += len(match["tp_indices"])

    model.roi_heads.score_thresh = orig
    return total_tp / max(total_gt, 1)
