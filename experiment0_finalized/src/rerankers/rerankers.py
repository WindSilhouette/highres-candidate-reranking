"""
rerankers.py
------------
All non-CARD reranking methods operating on frozen embeddings + logits.

Each reranker receives a patient group:
    embeddings : (n_les, D)
    abs_probs  : (n_les,)   calibrated absolute-risk probabilities
Returns:
    scores     : (n_les,)   HIGHER = more suspicious / anomalous

Score direction contract
------------------------
ALL rerankers must return scores where higher == more suspicious.
Unsupervised relative scorers (centroid, kNN, TOAR-lite) use
`apply_score_flip_if_needed` after inference to verify and
auto-correct direction on the training/val set.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ── Score direction sanity check ──────────────────────────────────────────────

def apply_score_flip_if_needed(
    scores: np.ndarray,
    labels: np.ndarray,
    name: str = "reranker",
    verbose: bool = True,
) -> tuple:
    """
    Checks whether scores are positively or negatively correlated with labels.
    If AUROC < 0.5, flips the scores (multiply by -1) so that
    higher score == more suspicious.

    Parameters
    ----------
    scores : (N,) flat scores across all patients
    labels : (N,) binary labels
    name   : reranker name for logging

    Returns
    -------
    (scores, flipped: bool, auroc_before: float)
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return scores, False, 0.5

    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        return scores, False, 0.5

    flipped = False
    if auroc < 0.5:
        scores = -scores
        flipped = True
        if verbose:
            print(f"  ⚠️  [{name}] Score direction flipped "
                  f"(AUROC was {auroc:.3f}, < 0.5). "
                  f"New AUROC ≈ {1-auroc:.3f}")
    else:
        if verbose:
            print(f"  ✓  [{name}] Score direction OK (AUROC={auroc:.3f})")
    return scores, flipped, auroc


# ── 1. Absolute-risk only ─────────────────────────────────────────────────────

class AbsoluteRiskReranker:
    """Baseline: just use the calibrated classifier probability."""
    name = "absolute_risk"
    is_trainable = False

    def score(self, embeddings: np.ndarray, abs_probs: np.ndarray,
              **kwargs) -> np.ndarray:
        return abs_probs.copy()


# ── 2. Centroid / Ugly-Duckling distance ──────────────────────────────────────

class CentroidReranker:
    """
    score_i = distance(embedding_i, patient_centroid).
    Higher = more unusual relative to patient mean.
    """
    is_trainable = False

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.name = f"centroid_{metric}"

    def score(self, embeddings: np.ndarray, abs_probs: np.ndarray,
              **kwargs) -> np.ndarray:
        centroid = embeddings.mean(axis=0)
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            cnorm = np.linalg.norm(centroid) + 1e-8
            cos_sim = (embeddings @ centroid) / (norms.squeeze() * cnorm)
            return 1.0 - cos_sim          # higher = more different from centroid
        else:
            diffs = embeddings - centroid[None, :]
            return np.linalg.norm(diffs, axis=1)


# ── 3. kNN / Prototype distance ───────────────────────────────────────────────

class KNNReranker:
    """
    score_i = mean distance to k nearest neighbours within the patient set.
    Outlier lesion has high distance to all neighbours.
    """
    is_trainable = False

    def __init__(self, k: int = 3):
        self.k = k
        self.name = f"knn_k{k}"

    def score(self, embeddings: np.ndarray, abs_probs: np.ndarray,
              **kwargs) -> np.ndarray:
        n = len(embeddings)
        k = min(self.k, n - 1)
        if k == 0:
            return np.zeros(n)
        diffs = embeddings[:, None, :] - embeddings[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)
        return sorted_dists[:, :k].mean(axis=1)


# ── 4. TOAR-Lite ──────────────────────────────────────────────────────────────

class TOARLite(nn.Module):
    """
    Trainable per-patient anomaly scorer.
      1. Population normalisation (subtract patient mean)
      2. Leave-one-out context: mean of all other lesions
      3. Residual deviation d_i = norm_emb_i - context_i
      4. Score = ||d_i|| + MLP(d_i)

    Score direction: geometric deviation is always non-negative, but the
    MLP component can invert direction during training on small datasets.
    Auto-flip is applied post-training via apply_score_flip_if_needed.
    """
    name = "toar_lite"
    is_trainable = True

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learned flip scalar initialised to +1
        # Allows the network to express "high deviation = benign" without
        # the geometric term overpowering the learned signal.
        self.direction = nn.Parameter(torch.ones(1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings : (n, D) → scores : (n,)"""
        n, D = embeddings.shape
        mu    = embeddings.mean(dim=0, keepdim=True)
        sigma = embeddings.std(dim=0, keepdim=True) + 1e-6
        norm_emb = (embeddings - mu) / sigma

        sum_all = norm_emb.sum(dim=0, keepdim=True)
        context  = (sum_all - norm_emb) / max(n - 1, 1)

        deviation  = norm_emb - context
        geom_score = deviation.norm(dim=1)
        mlp_score  = self.mlp(deviation).squeeze(-1)
        raw_score  = geom_score + mlp_score
        # direction parameter lets the network learn the sign
        return raw_score * self.direction

    @torch.no_grad()
    def score(self, embeddings: np.ndarray, abs_probs: np.ndarray,
              **kwargs) -> np.ndarray:
        self.eval()
        t = torch.from_numpy(embeddings).float()
        return self.forward(t).numpy()


# ── 5. Simple Set Transformer ─────────────────────────────────────────────────

class SetTransformerReranker(nn.Module):
    """
    Small set transformer: projection → L × (MultiheadSelfAttention + FFN)
    → per-lesion score head.
    """
    name = "set_transformer"
    is_trainable = True

    def __init__(self, embedding_dim: int, n_heads: int = 2,
                 n_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(embedding_dim, ff_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim, nhead=n_heads,
            dim_feedforward=ff_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.score_head = nn.Linear(ff_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings : (n, D) → scores : (n,)"""
        x = self.input_proj(embeddings).unsqueeze(0)
        x = self.transformer(x)
        return self.score_head(x).squeeze(0).squeeze(-1)

    @torch.no_grad()
    def score(self, embeddings: np.ndarray, abs_probs: np.ndarray,
              **kwargs) -> np.ndarray:
        self.eval()
        t = torch.from_numpy(embeddings).float()
        return self.forward(t).numpy()


# ── Registry ──────────────────────────────────────────────────────────────────

def get_non_trainable_rerankers() -> List:
    return [
        AbsoluteRiskReranker(),
        CentroidReranker(metric="euclidean"),
        CentroidReranker(metric="cosine"),
        KNNReranker(k=3),
        KNNReranker(k=5),
    ]
