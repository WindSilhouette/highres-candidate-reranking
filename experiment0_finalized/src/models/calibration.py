"""
calibration.py
--------------
Post-hoc probability calibration using validation patients only.
Methods: temperature scaling (preferred) or Platt scaling.
Reports ECE and Brier score before and after calibration.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# ── Temperature scaling ────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Single scalar temperature T > 0.
    calibrated_prob = sigmoid(logit / T)
    Fit by minimising NLL on val set.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature.clamp(min=0.01))

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> float:
        logits_t = torch.from_numpy(val_logits).float()
        labels_t = torch.from_numpy(val_labels).float()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr,
                                       max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature.clamp(min=0.01)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(self.temperature.item())
        print(f"  Temperature scaling: T = {T:.4f}")
        return T

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(logits).float()
            return self(t).numpy()


# ── Platt scaling ─────────────────────────────────────────────────────────────

class PlattScaling:
    """
    Logistic regression on logits (A * logit + B).
    Fit on validation patients only.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        self.model.fit(val_logits.reshape(-1, 1), val_labels.astype(int))

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]


# ── Metrics ───────────────────────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ── Main calibration pipeline ─────────────────────────────────────────────────

def calibrate(
    raw_logits: dict,           # keys: patient_id → {logits, labels}
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    all_logits: np.ndarray,
    all_labels: np.ndarray,
    method: str = "temperature",
    report_path: str = "outputs/calibration_report.json",
) -> np.ndarray:
    """
    Fit calibration on val_logits/val_labels.
    Apply to all_logits.
    Returns calibrated probabilities for all lesions.
    """
    raw_probs_val = 1 / (1 + np.exp(-val_logits))
    ece_before = expected_calibration_error(raw_probs_val, val_labels)
    brier_before = brier_score(raw_probs_val, val_labels)

    if method == "temperature":
        calibrator = TemperatureScaling()
        T = calibrator.fit(val_logits, val_labels)
        cal_probs_all = calibrator.calibrate(all_logits)
        cal_probs_val = calibrator.calibrate(val_logits)
        extra = {"temperature": T}
    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(val_logits, val_labels)
        cal_probs_all = calibrator.calibrate(all_logits)
        cal_probs_val = calibrator.calibrate(val_logits)
        extra = {}
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    ece_after = expected_calibration_error(cal_probs_val, val_labels)
    brier_after = brier_score(cal_probs_val, val_labels)

    report = {
        "method": method,
        "val_ece_before": round(ece_before, 4),
        "val_ece_after":  round(ece_after,  4),
        "val_brier_before": round(brier_before, 4),
        "val_brier_after":  round(brier_after,  4),
        **extra,
    }

    print(f"\n[Calibration Report — {method}]")
    print(f"  ECE   : {ece_before:.4f} → {ece_after:.4f}")
    print(f"  Brier : {brier_before:.4f} → {brier_after:.4f}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(f"  Saved: {report_path}")

    return cal_probs_all, report
