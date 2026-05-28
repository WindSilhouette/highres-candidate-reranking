"""
src/detector/toy_images.py
--------------------------
Generates synthetic skin-region images with circular lesion annotations.
Images are realistic enough to test the full detector pipeline without
requiring real iToBoS/ISIC data.

Skin texture: Gaussian noise + low-frequency variation on warm skin tone.
Lesions: darker ellipses with slightly irregular borders + subtle colour shift.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


# ── Skin-like background generator ───────────────────────────────────────────

def _make_skin_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Warm, slightly mottled skin tone background."""
    # Base skin tone (R, G, B) — varies by patient phototype
    base_r = int(rng.integers(170, 220))
    base_g = int(rng.integers(120, 160))
    base_b = int(rng.integers(90, 130))

    # Low-frequency variation (large Gaussian blobs)
    lf = rng.normal(0, 10, (h // 8, w // 8, 3)).astype(np.float32)
    lf = np.array(
        Image.fromarray(
            np.clip(lf + 128, 0, 255).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) - 128

    # High-frequency noise (skin texture)
    hf = rng.normal(0, 4, (h, w, 3)).astype(np.float32)

    img = np.stack([
        np.clip(base_r + lf[:, :, 0] + hf[:, :, 0], 0, 255),
        np.clip(base_g + lf[:, :, 1] + hf[:, :, 1], 0, 255),
        np.clip(base_b + lf[:, :, 2] + hf[:, :, 2], 0, 255),
    ], axis=2).astype(np.uint8)
    return img


# ── Lesion painter ────────────────────────────────────────────────────────────

def _paint_lesion(
    img: np.ndarray,
    cx: int, cy: int,
    rx: int, ry: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Paint one synthetic lesion (dark ellipse with irregular border).
    Returns modified image and tight bounding box (x1, y1, x2, y2).
    """
    h, w = img.shape[:2]
    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Lesion colour: darker + slightly different hue than skin
    dark  = int(rng.integers(60, 120))  # increased for better contrast
    r_col = max(0, int(img[cy, cx, 0]) - dark - int(rng.integers(0, 20)))
    g_col = max(0, int(img[cy, cx, 1]) - dark)
    b_col = max(0, int(img[cy, cx, 2]) - dark + int(rng.integers(0, 15)))
    colour = (r_col, g_col, b_col)

    # Draw main ellipse
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=colour,
        outline=None,
    )

    # Slightly irregular border (3 random extra spots)
    for _ in range(3):
        bx = cx + int(rng.integers(-rx, rx))
        by = cy + int(rng.integers(-ry, ry))
        br = int(rng.integers(1, max(2, min(rx, ry) // 3)))
        draw.ellipse([bx - br, by - br, bx + br, by + br],
                     fill=colour)

    # Smooth slightly
    pil   = pil.filter(ImageFilter.GaussianBlur(radius=1))
    img   = np.array(pil)

    x1 = max(0, cx - rx - 2)
    y1 = max(0, cy - ry - 2)
    x2 = min(w - 1, cx + rx + 2)
    y2 = min(h - 1, cy + ry + 2)
    return img, (x1, y1, x2, y2)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_toy_detection_dataset(
    n_images: int = 80,
    image_h: int = 256,
    image_w: int = 256,
    lesions_per_image_min: int = 1,
    lesions_per_image_max: int = 6,
    lesion_radius_min: int = 8,
    lesion_radius_max: int = 28,
    n_patients: int = 20,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    output_dir: str = "detector_outputs/toy_images",
    annotations_csv: str = "detector_outputs/toy_annotations.csv",
) -> pd.DataFrame:
    """
    Generate synthetic skin images with lesion bounding boxes.

    Returns
    -------
    df : DataFrame with one row per annotation box.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(annotations_csv).parent.mkdir(parents=True, exist_ok=True)

    patients = [f"PAT_{i:03d}" for i in range(n_patients)]

    # Split images by patient-image assignment
    n_train = int(n_images * train_frac)
    n_val   = int(n_images * val_frac)
    splits  = (["train"] * n_train + ["val"] * n_val +
               ["test"] * (n_images - n_train - n_val))
    random.shuffle(splits)

    rows = []
    for img_idx in range(n_images):
        patient_id = patients[img_idx % n_patients]
        split      = splits[img_idx]
        image_id   = f"IMG_{img_idx:04d}"
        img_path   = out / f"{image_id}.png"

        # Generate skin background
        img = _make_skin_background(image_h, image_w, rng)

        # Generate lesions
        n_les = int(rng.integers(lesions_per_image_min,
                                  lesions_per_image_max + 1))
        placed = []
        attempts = 0
        while len(placed) < n_les and attempts < 200:
            attempts += 1
            rx = int(rng.integers(lesion_radius_min, lesion_radius_max))
            ry = int(rng.integers(lesion_radius_min, lesion_radius_max))
            cx = int(rng.integers(rx + 4, image_w - rx - 4))
            cy = int(rng.integers(ry + 4, image_h - ry - 4))

            # Avoid overlap with already placed lesions
            overlap = False
            for (px, py, prx, pry) in placed:
                if abs(cx - px) < rx + prx + 4 and abs(cy - py) < ry + pry + 4:
                    overlap = True
                    break
            if overlap:
                continue

            img, bbox = _paint_lesion(img, cx, cy, rx, ry, rng)
            placed.append((cx, cy, rx, ry))

            rows.append({
                "image_path": str(img_path),
                "image_id":   image_id,
                "patient_id": patient_id,
                "split":      split,
                "bbox_x1":    bbox[0],
                "bbox_y1":    bbox[1],
                "bbox_x2":    bbox[2],
                "bbox_y2":    bbox[3],
                "class_label": "lesion",
                "class_id":    1,
            })

        # Save image
        Image.fromarray(img).save(img_path)

    df = pd.DataFrame(rows)
    df.to_csv(annotations_csv, index=False)

    print(f"\n[Toy Detection Dataset]")
    print(f"  Images     : {n_images}")
    print(f"  Annotations: {len(df)}")
    print(f"  Patients   : {n_patients}")
    for s in ["train", "val", "test"]:
        n = df[df["split"] == s]["image_id"].nunique()
        print(f"  {s:5s} images: {n}")
    print(f"  Saved to   : {output_dir}")
    return df
