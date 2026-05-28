"""
plots.py
--------
All output plots for Experiment 0.
  1. SE@k curve
  2. Precision@k curve
  3. Calibration curve (reliability diagram)
  4. NNT comparison bar chart
  5. Score distribution histograms (positive vs negative per reranker)
  6. CARD ablation table (saved as PNG)
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


COLORS = [
    "#1B3A6B", "#C0392B", "#1A6B3C", "#8E44AD",
    "#E67E22", "#2980B9", "#27AE60", "#884444",
    "#7F8C8D", "#F39C12", "#16A085", "#D35400",
]


def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ── 1. SE@k curve ─────────────────────────────────────────────────────────────

def plot_se_at_k(results: dict, k_values: List[int], output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, m) in enumerate(results.items()):
        vals = [m.get(f"SE@{k}") for k in k_values]
        ci   = [m.get(f"SE@{k}_ci95", [None, None]) for k in k_values]
        valid_k = [k for k, v in zip(k_values, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        if not valid_k:
            continue
        color = COLORS[i % len(COLORS)]
        ax.plot(valid_k, valid_v, marker="o", color=color,
                label=name, linewidth=1.8, markersize=5)
        # 95% CI band
        lo_vals = [ci[j][0] for j, k in enumerate(k_values)
                   if k in valid_k and ci[j][0] is not None]
        hi_vals = [ci[j][1] for j, k in enumerate(k_values)
                   if k in valid_k and ci[j][1] is not None]
        if len(lo_vals) == len(valid_k):
            ax.fill_between(valid_k, lo_vals, hi_vals,
                            alpha=0.10, color=color)

    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.9,
               label="80% target")
    ax.axhline(0.9, color="grey", linestyle=":",  linewidth=0.9,
               label="90% target")
    ax.set_xlabel("k (lesions reviewed per patient)")
    ax.set_ylabel("Sensitivity@k  (positive patients only)")
    ax.set_title("SE@k curve with 95% bootstrap CI")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _save(fig, output_path)


# ── 2. Precision@k curve ──────────────────────────────────────────────────────

def plot_precision_at_k(results: dict, k_values: List[int], output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, m) in enumerate(results.items()):
        vals = [m.get(f"P@{k}_all_patients", 0) for k in k_values]
        ax.plot(k_values, vals, marker="s", color=COLORS[i % len(COLORS)],
                label=name, linewidth=1.8, markersize=5)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k  (all patients)")
    ax.set_title("Precision@k curve")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _save(fig, output_path)


# ── 3. Calibration curve ──────────────────────────────────────────────────────

def plot_calibration_curve(probs_raw, probs_cal, labels, output_path,
                            n_bins=10):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=1)
    for probs, label, color in [
        (probs_raw, "Uncalibrated", "#C0392B"),
        (probs_cal, "Calibrated",   "#1A6B3C"),
    ]:
        bins = np.linspace(0, 1, n_bins + 1)
        fp, mp = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            fp.append(labels[mask].mean())
            mp.append(probs[mask].mean())
        ax.plot(mp, fp, marker="o", color=color,
                label=label, linewidth=1.8, markersize=5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    _save(fig, output_path)


# ── 4. NNT comparison ─────────────────────────────────────────────────────────

def plot_nnt_comparison(results, output_path,
                         sensitivity_targets=(0.80, 0.90)):
    n_targets = len(sensitivity_targets)
    fig, axes = plt.subplots(1, n_targets,
                              figsize=(6 * n_targets, 5), sharey=False)
    if n_targets == 1:
        axes = [axes]
    names = list(results.keys())
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

    for ax, target in zip(axes, sensitivity_targets):
        key  = f"NNT@{int(target*100)}%sens"
        vals = [results[n].get(key) for n in names]
        bar_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(range(len(names)), bar_vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=38, ha="right", fontsize=7)
        ax.set_ylabel("Mean lesions reviewed (NNT)")
        ax.set_title(f"NNT at {int(target*100)}% sensitivity\n"
                     f"(positive patients only)")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    _save(fig, output_path)


# ── 5. Score distribution histograms ──────────────────────────────────────────

def plot_score_distributions(
    patient_groups: List[dict],
    scores_dict: Dict[str, np.ndarray],
    output_dir: str,
    max_plots: int = 12,
):
    """
    For each reranker: histogram of scores for positive vs negative lesions.
    Helps visually verify score direction and separation.
    """
    # Build flat labels
    all_labels = np.concatenate([
        (grp["labels"].numpy()
         if hasattr(grp["labels"], "numpy")
         else np.array(grp["labels"]))
        for grp in patient_groups
    ])

    names = list(scores_dict.keys())[:max_plots]
    n_cols = 3
    n_rows = (len(names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten() if n_rows > 1 else np.array([axes]).flatten()

    for ax, name in zip(axes, names):
        scores = scores_dict[name]
        pos_scores = scores[all_labels == 1]
        neg_scores = scores[all_labels == 0]

        bins = np.linspace(scores.min(), scores.max(), 30)
        ax.hist(neg_scores, bins=bins, alpha=0.6, color="#2980B9",
                label=f"Benign (n={len(neg_scores)})", density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.8, color="#C0392B",
                label=f"Malignant (n={len(pos_scores)})", density=True)

        # AUROC annotation
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, scores)
            direction = "↑ correct" if auc >= 0.5 else "↓ FLIPPED"
            ax.set_title(f"{name}\nAUROC={auc:.3f} {direction}", fontsize=8)
        except Exception:
            ax.set_title(name, fontsize=8)

        ax.legend(fontsize=6)
        ax.set_xlabel("Score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle("Score distributions: positive vs negative lesions\n"
                 "(higher score should = more suspicious)", fontsize=10)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save(fig, str(Path(output_dir) / "score_distributions.png"))


# ── 6. CARD ablation table (PNG) ─────────────────────────────────────────────

def plot_card_ablation_table(results: dict, output_path: str,
                              k_values=(1, 3, 5, 10, 15)):
    card_keys = [k for k in results if k.startswith("card_")]
    if not card_keys:
        return

    rows  = []
    cols  = (["Variant"] + [f"SE@{k}" for k in k_values] +
             ["AUROC", "MRR", "NNT@80", "NNT@90",
              "CandRed@80%", "CandRed@90%"])

    for key in card_keys:
        m    = results[key]
        name = key.replace("card_", "")
        def f(v): return f"{v:.3f}" if v is not None else "N/A"
        row  = [name]
        for k in k_values:
            row.append(f(m.get(f"SE@{k}")))
        row.append(f(m.get("AUROC")))
        row.append(f(m.get("MRR")))
        row.append(f(m.get("NNT@80%sens")))
        row.append(f(m.get("NNT@90%sens")))
        row.append(f(m.get("candidate_reduction@80%sens")))
        row.append(f(m.get("candidate_reduction@90%sens")))
        rows.append(row)

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.1),
                                    1.2 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Style header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1B3A6B")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor("#EBF3FB" if i % 2 == 0 else "white")

    ax.set_title("CARD Ablation Table\n"
                 "(metrics on test set, positive patients for SE/MRR/NNT)",
                 fontsize=11, pad=12)
    _save(fig, output_path)
    print(f"  CARD ablation table saved: {output_path}")
