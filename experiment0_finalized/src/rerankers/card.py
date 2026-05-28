"""
card.py
-------
CARD: Calibrated Anomaly Reranking by Deviation

A SMALL fusion model combining:
  - Absolute risk (calibrated classifier probability)
  - Relative anomaly (centroid deviation score)
  - Conflict (disagreement between absolute and relative signals)

Ablations (controlled by `variant`):
  "abs_only"         : input = [abs_prob]
  "rel_only"         : input = [rel_score]
  "abs_rel"          : input = [abs_prob, rel_score]
  "abs_rel_conflict" : input = [abs_prob, rel_score, conflict]

Architecture: tiny MLP (input → 32 → 16 → 1).
Trained per-patient (patient-batch episodes).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


CARD_VARIANTS = ["abs_only", "rel_only", "abs_rel", "abs_rel_conflict"]

_INPUT_DIMS = {
    "abs_only":         1,
    "rel_only":         1,
    "abs_rel":          2,
    "abs_rel_conflict": 3,
}


# ── CARD model (tiny) ─────────────────────────────────────────────────────────

class CARDModel(nn.Module):
    """
    Tiny fusion MLP.
    Must stay small — no opaque deep networks.
    """

    def __init__(self, variant: str, hidden_dim: int = 32):
        super().__init__()
        assert variant in CARD_VARIANTS, f"Unknown variant: {variant}"
        self.variant = variant
        in_dim = _INPUT_DIMS[variant]

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_features(
        self,
        abs_probs: torch.Tensor,    # (n,) calibrated absolute risk
        embeddings: torch.Tensor,   # (n, D)
    ) -> torch.Tensor:
        """Construct the input feature vector based on variant."""

        # Relative anomaly: centroid distance (always computed as intermediate)
        centroid = embeddings.mean(dim=0, keepdim=True)
        diffs = embeddings - centroid
        rel_score = diffs.norm(dim=1)                       # (n,)
        # Normalise to [0,1] within patient
        rel_score = (rel_score - rel_score.min()) / (
            rel_score.max() - rel_score.min() + 1e-8
        )

        if self.variant == "abs_only":
            return abs_probs.unsqueeze(-1)                  # (n, 1)

        elif self.variant == "rel_only":
            return rel_score.unsqueeze(-1)                  # (n, 1)

        elif self.variant == "abs_rel":
            return torch.stack([abs_probs, rel_score], dim=1)  # (n, 2)

        elif self.variant == "abs_rel_conflict":
            # Conflict: how much do abs and rel disagree?
            # High conflict = abs says low risk but rel says very different,
            # or vice versa.
            conflict = (abs_probs - rel_score).abs()        # (n,)
            return torch.stack(
                [abs_probs, rel_score, conflict], dim=1     # (n, 3)
            )

    def forward(
        self,
        abs_probs: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        abs_probs  : (n,)
        embeddings : (n, D)
        returns    : (n,)  final fusion score
        """
        features = self._build_features(abs_probs, embeddings)
        return self.net(features).squeeze(-1)               # (n,)

    @torch.no_grad()
    def score(
        self,
        embeddings: np.ndarray,
        abs_probs: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        self.eval()
        emb_t  = torch.from_numpy(embeddings).float()
        prob_t = torch.from_numpy(abs_probs).float()
        return self.forward(prob_t, emb_t).numpy()

    @property
    def name(self):
        return f"card_{self.variant}"


# ── Training ──────────────────────────────────────────────────────────────────

def train_card(
    variant: str,
    patient_groups_train: List[dict],
    patient_groups_val: List[dict],
    cal_probs: np.ndarray,
    all_lesion_ids: List[str],
    cfg: dict,
    checkpoint_dir: str = "outputs/card_checkpoints",
) -> CARDModel:
    """
    Train one CARD variant on patient episodes.
    Each episode = one patient's lesions.

    Loss: pairwise ranking loss.
    For each patient, malignant lesions should rank above benign ones.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint_dir) / f"card_{variant}_best.pt"
    device = torch.device(cfg.get("device", "cpu"))

    model = CARDModel(
        variant=variant,
        hidden_dim=cfg["rerankers"]["card_hidden"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["rerankers"]["card_lr"],
    )

    # Build lookup: lesion_id → calibrated prob
    lid2prob = {lid: prob for lid, prob in zip(all_lesion_ids, cal_probs)}

    best_val = float("inf")
    epochs = cfg["rerankers"]["card_epochs"]

    print(f"\n[CARD Training — variant: {variant}]")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = _run_epoch(model, patient_groups_train, lid2prob,
                                optimizer, device, training=True)
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, patient_groups_val, lid2prob,
                                  None, device, training=False)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} "
                  f"| val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Best val loss: {best_val:.4f} — saved to {ckpt_path}")
    return model


def _pairwise_ranking_loss(scores: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
    """
    For every (malignant, benign) pair in the patient,
    penalise if benign scores higher than malignant.
    Returns mean loss over all pairs.
    If no positive or no negative in patient, returns 0.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0)

    pos_scores = scores[pos_mask]   # (n_pos,)
    neg_scores = scores[neg_mask]   # (n_neg,)

    # All pairs: (n_pos, n_neg)
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    loss = F.softplus(-diff)       # log(1 + exp(-diff))
    return loss.mean()


def _run_epoch(
    model, patient_groups, lid2prob, optimizer, device, training: bool
) -> float:
    total_loss = 0.0
    n_patients = 0

    for grp in patient_groups:
        emb    = grp["embeddings"].to(device)
        labels = grp["labels"].to(device)
        lids   = grp["lesion_ids"]

        # Build abs_probs for this patient from calibrated lookup
        abs_probs = torch.tensor(
            [lid2prob.get(lid, 0.5) for lid in lids],
            dtype=torch.float32,
            device=device,
        )

        scores = model(abs_probs, emb)
        loss   = _pairwise_ranking_loss(scores, labels)

        if training and loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_patients += 1

    return total_loss / max(n_patients, 1)


def train_all_card_variants(
    patient_groups_train: List[dict],
    patient_groups_val: List[dict],
    cal_probs: np.ndarray,
    all_lesion_ids: List[str],
    cfg: dict,
) -> Dict[str, CARDModel]:
    """Train all four CARD variants and return dict of trained models."""
    trained = {}
    for variant in CARD_VARIANTS:
        model = train_card(
            variant=variant,
            patient_groups_train=patient_groups_train,
            patient_groups_val=patient_groups_val,
            cal_probs=cal_probs,
            all_lesion_ids=all_lesion_ids,
            cfg=cfg,
        )
        trained[variant] = model
    return trained
