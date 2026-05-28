"""
classifier.py
-------------
Independent lesion classifier (baseline).
Toy mode  : MLP on pre-computed embeddings.
Real mode : ResNet50 or EfficientNet-B0 on lesion crop images.

Saves: logits, probabilities, and penultimate embeddings for every lesion.
These frozen outputs feed all downstream rerankers.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LesionEmbeddingDataset


# ── Model definitions ─────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Simple MLP for toy (embedding) mode.
    Also serves as the classification head for real-image backbones.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(hidden_dim // 2, 1)
        self.embedding_dim = hidden_dim // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_layer(x)
        logit = self.classifier_head(features).squeeze(-1)
        return logit, features  # (B,), (B, emb_dim)


# ── Training ──────────────────────────────────────────────────────────────────

class ClassifierTrainer:

    def __init__(self, cfg: dict, embedding_dim: int):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model = MLPClassifier(
            input_dim=embedding_dim,
            hidden_dim=cfg["classifier"]["hidden_dim"],
            dropout=cfg["classifier"]["dropout"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["classifier"]["lr"],
            weight_decay=cfg["classifier"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = cfg["classifier"]["early_stopping_patience"]

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            x = batch["embedding"].to(self.device)
            y = batch["label"].float().to(self.device)
            self.optimizer.zero_grad()
            logit, _ = self.model(x)
            loss = self.criterion(logit, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            x = batch["embedding"].to(self.device)
            y = batch["label"].float().to(self.device)
            logit, _ = self.model(x)
            loss = self.criterion(logit, y)
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    def fit(
        self,
        train_ds: LesionEmbeddingDataset,
        val_ds: LesionEmbeddingDataset,
        checkpoint_path: str,
    ):
        bs = self.cfg["classifier"]["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

        print("\n[Classifier Training]")
        for epoch in range(1, self.cfg["classifier"]["epochs"] + 1):
            tr_loss = self.train_epoch(train_loader)
            va_loss = self.val_epoch(val_loader)
            self.scheduler.step(va_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | train_loss={tr_loss:.4f} "
                      f"| val_loss={va_loss:.4f}")

            if va_loss < self.best_val_loss:
                self.best_val_loss = va_loss
                self.patience_counter = 0
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        print(f"  Best val_loss: {self.best_val_loss:.4f}")


# ── Inference / embedding extraction ─────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: MLPClassifier,
    ds: LesionEmbeddingDataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """
    Run full dataset through classifier and collect:
    - logits      : (N,)
    - probs       : (N,)  sigmoid of logits
    - embeddings  : (N, D)  penultimate layer features
    - labels      : (N,)
    - patient_ids : list[str]
    - lesion_ids  : list[str]
    """
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits, all_probs, all_embs = [], [], []
    all_labels, all_pids, all_lids = [], [], []

    for batch in tqdm(loader, desc="Extracting embeddings"):
        x = batch["embedding"].to(device)
        logit, emb = model(x)
        prob = torch.sigmoid(logit)
        all_logits.append(logit.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
        all_embs.append(emb.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_pids.extend(batch["patient_id"])
        all_lids.extend(batch["lesion_id"])

    return {
        "logits":      np.concatenate(all_logits),
        "probs":       np.concatenate(all_probs),
        "embeddings":  np.concatenate(all_embs),
        "labels":      np.concatenate(all_labels),
        "patient_ids": all_pids,
        "lesion_ids":  all_lids,
    }


def save_embeddings(result: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        logits=result["logits"],
        probs=result["probs"],
        embeddings=result["embeddings"],
        labels=result["labels"],
        patient_ids=np.array(result["patient_ids"]),
        lesion_ids=np.array(result["lesion_ids"]),
    )
    print(f"  Embeddings saved to {path}")


def load_embeddings(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "logits":      d["logits"],
        "probs":       d["probs"],
        "embeddings":  d["embeddings"],
        "labels":      d["labels"],
        "patient_ids": d["patient_ids"].tolist(),
        "lesion_ids":  d["lesion_ids"].tolist(),
    }
