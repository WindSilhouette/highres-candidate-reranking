"""
trainer.py
----------
Training loops for TOAR-lite and Set Transformer.
Includes post-training score direction verification and auto-flip.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rerankers.rerankers import (
    TOARLite, SetTransformerReranker, apply_score_flip_if_needed
)


# ── Pairwise ranking loss ─────────────────────────────────────────────────────

def pairwise_ranking_loss(scores: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
    """
    Penalise every (malignant, benign) pair where benign scores >= malignant.
    Returns mean softplus loss. Returns 0 if no paired examples exist.
    """
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return torch.zeros(1, requires_grad=scores.requires_grad)[0]
    pos_s = scores[pos]
    neg_s = scores[neg]
    diff  = pos_s.unsqueeze(1) - neg_s.unsqueeze(0)   # (n_pos, n_neg)
    return F.softplus(-diff).mean()


# ── Generic episode trainer ───────────────────────────────────────────────────

def train_reranker(
    model: nn.Module,
    patient_groups_train: List[dict],
    patient_groups_val: List[dict],
    lr: float,
    epochs: int,
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "reranker",
    # Post-training flip check
    run_flip_check: bool = True,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val   = float("inf")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training {model_name}]")
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = _run_epoch(model, patient_groups_train,
                             optimizer, device, training=True)
        model.eval()
        with torch.no_grad():
            va_loss = _run_epoch(model, patient_groups_val,
                                 None, device, training=False)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr_loss:.4f} | "
                  f"val={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"  Best val loss: {best_val:.4f}")

    # ── Score direction check on val patients ─────────────────────────────
    if run_flip_check:
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for grp in patient_groups_val:
                emb    = grp["embeddings"].to(device)
                labels = grp["labels"].numpy()
                sc     = model(emb).cpu().numpy()
                all_scores.append(sc)
                all_labels.append(labels)
        if all_scores:
            flat_s = np.concatenate(all_scores)
            flat_l = np.concatenate(all_labels)
            _, flipped, auroc_before = apply_score_flip_if_needed(
                flat_s, flat_l, name=model_name, verbose=True
            )
            # If flipped, store the flip flag on the model for inference
            model._score_flip = flipped
        else:
            model._score_flip = False
    else:
        model._score_flip = False

    return model


def _run_epoch(model, groups, optimizer, device, training):
    total = 0.0
    for grp in groups:
        emb    = grp["embeddings"].to(device)
        labels = grp["labels"].to(device)
        scores = model(emb)
        loss   = pairwise_ranking_loss(scores, labels)
        if training and hasattr(loss, "backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / max(len(groups), 1)


# ── Score method that respects flip flag ──────────────────────────────────────

def score_with_flip(model: nn.Module, embeddings: np.ndarray,
                    abs_probs: np.ndarray) -> np.ndarray:
    """Wrapper that applies the post-training flip flag if set."""
    raw = model.score(embeddings=embeddings, abs_probs=abs_probs)
    if getattr(model, "_score_flip", False):
        return -raw
    return raw


# ── Convenience wrappers ──────────────────────────────────────────────────────

def train_toar_lite(patient_groups_train, patient_groups_val,
                    embedding_dim, cfg, device) -> TOARLite:
    model = TOARLite(
        embedding_dim=embedding_dim,
        hidden_dim=cfg["rerankers"]["toar_lite_hidden"],
    ).to(device)
    return train_reranker(
        model=model,
        patient_groups_train=patient_groups_train,
        patient_groups_val=patient_groups_val,
        lr=cfg["rerankers"]["toar_lite_lr"],
        epochs=cfg["rerankers"]["toar_lite_epochs"],
        checkpoint_path=cfg["rerankers"]["toar_lite_checkpoint"],
        device=device,
        model_name="TOAR-Lite",
        run_flip_check=True,
    )


def train_set_transformer(patient_groups_train, patient_groups_val,
                           embedding_dim, cfg, device) -> SetTransformerReranker:
    model = SetTransformerReranker(
        embedding_dim=embedding_dim,
        n_heads=cfg["rerankers"]["set_transformer_heads"],
        n_layers=cfg["rerankers"]["set_transformer_layers"],
    ).to(device)
    return train_reranker(
        model=model,
        patient_groups_train=patient_groups_train,
        patient_groups_val=patient_groups_val,
        lr=cfg["rerankers"]["set_transformer_lr"],
        epochs=cfg["rerankers"]["set_transformer_epochs"],
        checkpoint_path=cfg["rerankers"]["set_transformer_checkpoint"],
        device=device,
        model_name="Set Transformer",
        run_flip_check=True,
    )
