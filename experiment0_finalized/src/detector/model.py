"""
src/detector/model.py
---------------------
Thin wrapper around torchvision Faster R-CNN.
Configured for HIGH RECALL — thesis goal is candidate generation,
not state-of-the-art mAP.

Key settings for high recall:
  - Very low NMS threshold (keep more overlapping boxes)
  - Very low score threshold at inference
  - More detections per image
  - RPN recall-tuned anchors

Also provides a feature extractor to pull RoI features for UMAP.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou


# ── Model factory ─────────────────────────────────────────────────────────────

def build_detector(
    model_name: str = "fasterrcnn_resnet50_fpn",
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    nms_thresh: float = 0.3,
    score_thresh: float = 0.01,
    detections_per_img: int = 100,
    device: str = "cpu",
) -> FasterRCNN:
    """
    Build a Faster R-CNN detector configured for high recall.

    Parameters
    ----------
    num_classes   : 2 = background + lesion
    nms_thresh    : lower = keep more overlapping boxes (more recall)
    score_thresh  : lower = keep more low-confidence boxes (more recall)
    detections_per_img : increase to catch many small lesions
    """
    weights = "DEFAULT" if pretrained_backbone else None

    if model_name == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=weights,
            # High-recall RPN settings
            rpn_pre_nms_top_n_train=4000,
            rpn_pre_nms_top_n_test=4000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_score_thresh=0.0,
            # ROI head settings
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
        )
    elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=weights,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Replace classification head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(torch.device(device))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, epoch: int, loss: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "loss":        loss,
    }, path)


def load_checkpoint(model: nn.Module, path: str,
                     device: torch.device) -> Tuple[nn.Module, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", 0)
    print(f"  Loaded checkpoint: epoch={epoch}, loss={ckpt.get('loss', 'N/A'):.4f}")
    return model, epoch


# ── Feature extractor (for UMAP) ─────────────────────────────────────────────

class ROIFeatureExtractor:
    """
    Extracts RoI-pooled features from Faster R-CNN backbone + FPN.
    Used to get per-candidate embeddings for UMAP visualisation.
    """

    def __init__(self, model: FasterRCNN, device: torch.device):
        self.model  = model
        self.device = device
        self._hooks  = []
        self._feats  = {}
        self._register_hooks()

    def _register_hooks(self):
        def _hook(name):
            def fn(module, inp, out):
                self._feats[name] = out.detach().cpu()
            return fn

        # Hook the flatten layer after RoI pooling
        h = self.model.roi_heads.box_head.register_forward_hook(
            _hook("roi_features")
        )
        self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def extract(self, images: List[torch.Tensor],
                 boxes_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Run one forward pass and return RoI features for given boxes.
        Returns tensor (N_boxes, feature_dim) or None if no boxes.
        """
        self.model.eval()
        self._feats.clear()
        images = [img.to(self.device) for img in images]

        # Temporarily lower score threshold to keep all boxes
        orig_thresh = self.model.roi_heads.score_thresh
        self.model.roi_heads.score_thresh = 0.0
        try:
            _ = self.model(images)
        except Exception:
            pass
        finally:
            self.model.roi_heads.score_thresh = orig_thresh

        feats = self._feats.get("roi_features", None)
        return feats  # may be None if hook didn't fire


# ── Box matching utility ──────────────────────────────────────────────────────

def match_predictions_to_gt(
    pred_boxes: torch.Tensor,   # (N, 4)
    pred_scores: torch.Tensor,  # (N,)
    gt_boxes: torch.Tensor,     # (M, 4)
    iou_thresh: float = 0.3,
) -> Dict[str, list]:
    """
    Match predicted boxes to ground-truth boxes.

    Returns dict with:
      tp_indices   : list of pred indices that match a GT
      fp_indices   : list of pred indices with no GT match
      missed_gt    : list of GT indices not matched by any pred
      matched_gt   : list of GT index matched to each pred (-1 if FP)
      ious         : list of best IoU for each pred
    """
    result = {
        "tp_indices": [],
        "fp_indices": [],
        "missed_gt":  [],
        "matched_gt": [-1] * len(pred_boxes),
        "ious":       [0.0] * len(pred_boxes),
    }

    if len(pred_boxes) == 0:
        result["missed_gt"] = list(range(len(gt_boxes)))
        return result

    if len(gt_boxes) == 0:
        result["fp_indices"] = list(range(len(pred_boxes)))
        return result

    ious = box_iou(pred_boxes, gt_boxes)   # (N, M)
    best_iou, best_gt = ious.max(dim=1)    # (N,)
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    # Sort preds by score descending (greedy matching)
    order = pred_scores.argsort(descending=True)
    for i in order.tolist():
        gt_i = best_gt[i].item()
        iou_i = best_iou[i].item()
        result["ious"][i] = iou_i

        if iou_i >= iou_thresh and not gt_matched[gt_i]:
            result["tp_indices"].append(i)
            result["matched_gt"][i] = gt_i
            gt_matched[gt_i] = True
        else:
            result["fp_indices"].append(i)

    result["missed_gt"] = torch.where(~gt_matched)[0].tolist()
    return result
