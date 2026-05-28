"""
src/detector/interpretability.py
---------------------------------
All visual diagnostics for the first-stage detector:
  1. Box overlay plots (pred + GT on image)
  2. Top FP gallery
  3. Top TP gallery
  4. Missed-lesion gallery
  5. Confidence histogram (TP vs FP)
  6. Threshold vs recall / FP-per-image plot (FROC-like)
  7. UMAP / t-SNE of candidate embeddings
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


COLORS = {"tp": "#1A6B3C", "fp": "#C0392B", "gt": "#1B3A6B", "missed": "#E67E22"}


# ── 1. Box overlay ────────────────────────────────────────────────────────────

def plot_box_overlays(
    pred_df: pd.DataFrame,
    threshold: float,
    output_dir: str,
    n_images: int = 8,
    iou_thresh: float = 0.3,
):
    """Save one overlay image per sample image (up to n_images)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    actual = pred_df[pred_df["score"] >= threshold]
    missed = pred_df[pred_df["score"] == -1.0]

    image_ids = actual["image_id"].unique()[:n_images]

    for img_id in image_ids:
        img_preds  = actual[actual["image_id"] == img_id]
        img_missed = missed[missed["image_id"] == img_id]

        if len(img_preds) == 0:
            continue

        img_path = img_preds["image_path"].iloc[0]
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)

        for _, row in img_preds.iterrows():
            color = COLORS["tp"] if row["is_tp"] else COLORS["fp"]
            label = f"{'TP' if row['is_tp'] else 'FP'} {row['score']:.2f}"
            _draw_box(ax, row, color, label)

        for _, row in img_missed.iterrows():
            _draw_box(ax, row, COLORS["missed"], "MISSED", linestyle="--")

        # Legend
        handles = [
            patches.Patch(color=COLORS["tp"],     label="True Positive"),
            patches.Patch(color=COLORS["fp"],     label="False Positive"),
            patches.Patch(color=COLORS["missed"], label="Missed GT"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
        ax.set_title(f"Image {img_id} | thresh={threshold}", fontsize=9)
        ax.axis("off")

        out_path = out / f"overlay_{img_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Box overlays saved to {output_dir}/")


def _draw_box(ax, row, color, label, linestyle="-"):
    x1, y1 = row["bbox_x1"], row["bbox_y1"]
    x2, y2 = row["bbox_x2"], row["bbox_y2"]
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=1.5, edgecolor=color,
        facecolor="none", linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(x1, y1 - 3, label,
            color=color, fontsize=6, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.5, pad=1))


# ── 2-4. Crop galleries ───────────────────────────────────────────────────────

def _make_gallery(
    crop_paths: List[str],
    scores: List[float],
    title: str,
    output_path: str,
    n: int = 16,
):
    """Arrange up to n crops in a grid."""
    crop_paths = crop_paths[:n]
    scores     = scores[:n]
    if not crop_paths:
        return

    cols = min(4, len(crop_paths))
    rows = (len(crop_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]

    for i, (ax, path, score) in enumerate(zip(axes, crop_paths, scores)):
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(np.array(img))
        except Exception:
            ax.set_facecolor("#eeeeee")
        sc_txt = f"{score:.2f}" if score >= 0 else "GT"
        ax.set_title(sc_txt, fontsize=8)
        ax.axis("off")

    for ax in axes[len(crop_paths):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gallery: {output_path}")


def save_galleries(
    cand_df: pd.DataFrame,
    gallery_dir: str,
    n: int = 16,
):
    """Save TP, FP, and missed-lesion galleries."""
    gdir = Path(gallery_dir)
    gdir.mkdir(parents=True, exist_ok=True)

    if len(cand_df) == 0:
        return

    tp  = cand_df[cand_df["is_true_positive"] == 1].sort_values(
        "detector_score", ascending=False
    )
    fp  = cand_df[(cand_df["is_true_positive"] == 0) &
                  (cand_df["is_missed_gt"] == 0)].sort_values(
        "detector_score", ascending=False
    )
    mis = cand_df[cand_df["is_missed_gt"] == 1]

    _make_gallery(
        tp["candidate_image_path"].tolist(),
        tp["detector_score"].tolist(),
        "Top True Positives (by detector score)",
        str(gdir / "top_true_positives.png"), n,
    )
    _make_gallery(
        fp["candidate_image_path"].tolist(),
        fp["detector_score"].tolist(),
        "Top False Positives (by detector score) — candidates for FP analysis",
        str(gdir / "top_false_positives.png"), n,
    )
    _make_gallery(
        mis["candidate_image_path"].tolist(),
        [-1.0] * len(mis),
        "Missed Ground-Truth Lesions",
        str(gdir / "missed_lesions.png"), n,
    )


# ── 5. Confidence histogram ───────────────────────────────────────────────────

def plot_confidence_histogram(
    pred_df: pd.DataFrame,
    threshold: float,
    output_path: str,
):
    actual = pred_df[pred_df["score"] >= 0.0]
    if len(actual) == 0:
        return

    tp_scores = actual[actual["is_tp"] == 1]["score"].values
    fp_scores = actual[actual["is_fp"] == 1]["score"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(fp_scores, bins=bins, alpha=0.6, color=COLORS["fp"],
            label=f"False Positives (n={len(fp_scores)})", density=True)
    ax.hist(tp_scores, bins=bins, alpha=0.8, color=COLORS["tp"],
            label=f"True Positives (n={len(tp_scores)})", density=True)
    ax.axvline(threshold, color="black", linestyle="--",
               linewidth=1.2, label=f"Operating threshold ({threshold})")
    ax.set_xlabel("Detector confidence score")
    ax.set_ylabel("Density")
    ax.set_title("Confidence score distribution: TP vs FP")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confidence histogram: {output_path}")


# ── 6. Threshold vs recall / FP-per-image ────────────────────────────────────

def plot_froc_curve(
    sweep_df: pd.DataFrame,
    output_dir: str,
    operating_threshold: float = 0.1,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # FROC-like: recall vs FP/image
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["fp_per_image"], sweep_df["recall"],
            "o-", color=COLORS["tp"], linewidth=2, markersize=5)

    # Mark operating point
    op = sweep_df[sweep_df["threshold"].round(4) ==
                  round(operating_threshold, 4)]
    if len(op) > 0:
        ax.scatter(op["fp_per_image"], op["recall"],
                   color=COLORS["fp"], s=120, zorder=5,
                   label=f"Op. point (thresh={operating_threshold})")

    # Annotate each point with threshold
    for _, r in sweep_df.iterrows():
        ax.annotate(f"{r['threshold']:.2f}",
                    (r["fp_per_image"], r["recall"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="grey")

    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall (sensitivity)")
    ax.set_title("FROC-like curve: Recall vs FP/image")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.savefig(str(out / "froc_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Threshold vs recall and FP/image separately
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sweep_df["threshold"], sweep_df["recall"],
             "o-", color=COLORS["tp"], linewidth=2)
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Recall")
    ax1.set_title("Threshold vs Recall")
    ax1.axvline(operating_threshold, color="grey", linestyle="--",
                linewidth=1, label="Operating threshold")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(sweep_df["threshold"], sweep_df["fp_per_image"],
             "o-", color=COLORS["fp"], linewidth=2)
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("FP per image")
    ax2.set_title("Threshold vs FP/image")
    ax2.axvline(operating_threshold, color="grey", linestyle="--",
                linewidth=1, label="Operating threshold")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("Threshold sweep analysis", fontsize=11)
    fig.savefig(str(out / "fp_per_image_vs_recall.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  FROC and threshold plots saved to {output_dir}/")


# ── 7. UMAP / t-SNE of candidate embeddings ──────────────────────────────────

def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,   # 0=FP, 1=TP, 2=missed
    scores: np.ndarray,
    output_path: str,
    method: str = "tsne",
    n_components: int = 2,
    n_neighbors: int = 10,
    seed: int = 42,
):
    if len(embeddings) < 10:
        print("  Skipping embedding projection: too few samples")
        return

    print(f"  Running {method.upper()} on {len(embeddings)} candidates...")

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=seed,
            )
        except ImportError:
            print("  umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            random_state=seed,
            perplexity=min(30, max(5, len(embeddings) // 5)),
        )

    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {0: COLORS["fp"], 1: COLORS["tp"], 2: COLORS["missed"]}
    label_map = {0: "False Positive", 1: "True Positive", 2: "Missed GT"}

    for lab in [0, 1, 2]:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=color_map[lab], label=f"{label_map[lab]} (n={mask.sum()})",
            alpha=0.7, s=18, edgecolors="none",
        )

    ax.set_title(f"{method.upper()} of candidate embeddings")
    ax.legend(fontsize=8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Embedding plot: {output_path}")


# ── Print failure summary ─────────────────────────────────────────────────────

def print_failure_summary(pred_df: pd.DataFrame, threshold: float):
    actual = pred_df[pred_df["score"] >= threshold]
    missed = pred_df[pred_df["score"] == -1.0]

    tp_count = actual["is_tp"].sum()
    fp_count = actual["is_fp"].sum()
    ms_count = len(missed)
    total_gt = tp_count + ms_count

    print("\n[Failure Summary]")
    print(f"  GT lesions     : {total_gt}")
    print(f"  Detected (TP)  : {tp_count}  ({100*tp_count/max(total_gt,1):.1f}%)")
    print(f"  Missed         : {ms_count}  ({100*ms_count/max(total_gt,1):.1f}%)")
    print(f"  False Positives: {fp_count}")
    if tp_count + fp_count > 0:
        print(f"  Precision      : {100*tp_count/(tp_count+fp_count):.1f}%")
    print(f"\n  Motivation for Stage-2 reranker:")
    print(f"  → {fp_count} FPs per {actual['image_id'].nunique()} images "
          f"= {fp_count/max(actual['image_id'].nunique(),1):.1f} FP/image")
    print(f"  → Stage-2 goal: reduce FP/image while preserving recall")
