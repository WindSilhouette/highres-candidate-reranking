"""
src/embedding_extractor.py
==========================
Turn lesion images into per-lesion embeddings, row-aligned to metadata.parquet.

Real path : read images from the ISIC 2024 HDF5 by isic_id, run a pretrained
            ResNet50 or EfficientNet-B0 (torch/torchvision, imported lazily), and
            cache the result. An optional DINO/timm extractor sits behind
            use_dino and defaults OFF so the notebook never breaks without timm.
Synthetic : if no HDF5 is provided, synthesise embeddings that carry a realistic
            COMPLEMENTARY signal (a global classifier-learnable direction plus a
            per-patient outlier component, and some benign "nevus" outliers), so
            the downstream research engine is exercised meaningfully.

Caching   : if embeddings.npy + emb_index.json already exist and match the current
            metadata isic_id order, extraction is skipped.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from src.io_utils import read_table


def _cache_valid(emb_path, idx_path, isic_ids):
    if not (os.path.exists(emb_path) and os.path.exists(idx_path)):
        return False
    try:
        with open(idx_path) as f:
            cached = json.load(f)["isic_id"]
    except Exception:
        return False
    return cached == list(isic_ids)


# --------------------------------------------------------------------------- #
# Synthetic embeddings with complementary signal
# --------------------------------------------------------------------------- #
def _synthetic_embeddings(meta, dim, seed=0):
    rng = np.random.default_rng(seed)
    mal_dir = rng.normal(0, 1, dim); mal_dir /= np.linalg.norm(mal_dir)
    E = np.zeros((len(meta), dim), dtype=np.float32)
    centers = {pid: rng.normal(0, 1.5, dim) for pid in meta["patient_id"].unique()}
    for i, row in enumerate(meta.itertuples(index=False)):
        emb = centers[row.patient_id] + rng.normal(0, 1.0, dim)
        if row.malignant == 1:
            rdir = rng.normal(0, 1, dim); rdir /= np.linalg.norm(rdir)
            emb = emb + 2.2 * mal_dir + 2.0 * rdir            # signal + outlier
        elif rng.random() < 0.15:
            rdir = rng.normal(0, 1, dim); rdir /= np.linalg.norm(rdir)
            emb = emb + 2.5 * rdir                            # benign outlier (FP)
        E[i] = emb
    return E



# --------------------------------------------------------------------------- #
# Image path resolution for generic/iToBoS-style image folders
# --------------------------------------------------------------------------- #
def _candidate_paths(image_dir, token):
    token = str(token)
    if os.path.isabs(token):
        bases = [token]
    else:
        bases = [os.path.join(image_dir, token)]
    exts = ["", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    out = []
    for b in bases:
        root, ext = os.path.splitext(b)
        if ext:
            out.append(b)
        else:
            out.extend(root + e for e in exts if e)
    return out


def _resolve_image_paths(meta, image_dir):
    if not image_dir:
        return None
    ids = meta["isic_id"].astype(str).tolist()
    raw = meta["image_path"].astype(str).tolist() if "image_path" in meta.columns else ids
    paths, missing = [], []
    for isic, token in zip(ids, raw):
        found = None
        # Try the metadata path/filename first, then the canonical ID.
        for cand_token in [token, isic]:
            for cand in _candidate_paths(image_dir, cand_token):
                if os.path.exists(cand):
                    found = cand; break
            if found: break
        if found is None:
            missing.append(token)
            found = ""
        paths.append(found)
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise FileNotFoundError(f"Could not resolve {len(missing)} images under image_dir={image_dir}. Examples: {preview}")
    return paths

# --------------------------------------------------------------------------- #
# Real extractor (lazy torch/h5py imports; only used when hdf5 is provided)
# --------------------------------------------------------------------------- #
def _build_backbone(model_name, use_dino):
    import torch
    if use_dino:
        import timm  # optional; only imported when explicitly enabled
        net = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)
        feat_dim = net.num_features
    elif model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        net.classifier = torch.nn.Identity(); feat_dim = 1280
    else:  # resnet50 default
        from torchvision.models import resnet50, ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        net.fc = torch.nn.Identity(); feat_dim = 2048
    net.eval()
    return net, feat_dim


def _extract_image_paths(meta, cfg, paths):
    import torch
    from PIL import Image
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, feat_dim = _build_backbone(cfg["embedding"]["model"], cfg["embedding"]["use_dino"])
    net = net.to(device)
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    E = np.zeros((len(meta), feat_dim), dtype=np.float32)
    bs = cfg["embedding"]["batch_size"]
    with torch.no_grad():
        batch, rows = [], []
        def flush():
            if not batch:
                return
            x = torch.stack(batch).to(device)
            E[rows] = net(x).cpu().numpy()
            batch.clear(); rows.clear()
        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            batch.append(tfm(img)); rows.append(i)
            if len(batch) >= bs:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"    embedded {i+1}/{len(paths)}")
        flush()
    return E


def _extract_hdf5(meta, cfg, seed=0):
    import io
    import h5py
    import torch
    from PIL import Image
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, feat_dim = _build_backbone(cfg["embedding"]["model"], cfg["embedding"]["use_dino"])
    net = net.to(device)
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    E = np.zeros((len(meta), feat_dim), dtype=np.float32)
    bs = cfg["embedding"]["batch_size"]
    ids = meta["isic_id"].tolist()
    with h5py.File(cfg["paths"]["hdf5"], "r") as hf, torch.no_grad():
        batch, rows = [], []
        def flush():
            if not batch:
                return
            x = torch.stack(batch).to(device)
            E[rows] = net(x).cpu().numpy()
            batch.clear(); rows.clear()
        for i, isic in enumerate(ids):
            if isic not in hf:
                raise KeyError(f"Image key '{isic}' not found in HDF5. Check data.id_col matches the HDF5 keys.")
            img = Image.open(io.BytesIO(np.asarray(hf[isic]).tobytes())).convert("RGB")
            batch.append(tfm(img)); rows.append(i)
            if len(batch) >= bs:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"    embedded {i+1}/{len(ids)}")
        flush()
    return E


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract(cfg, seed=0):
    from src.config import artifact_paths
    ap = artifact_paths(cfg)
    meta = read_table(ap["metadata"])
    isic_ids = meta["isic_id"].tolist()

    if _cache_valid(ap["embeddings"], ap["emb_index"], isic_ids):
        print(f">> embeddings cache hit -> {ap['embeddings']} (skipping extraction)\n")
        return np.load(ap["embeddings"], mmap_mode="r")

    if cfg["paths"].get("hdf5") and os.path.exists(cfg["paths"]["hdf5"]):
        print(f">> extracting embeddings with {cfg['embedding']['model']} "
              f"(use_dino={cfg['embedding']['use_dino']}) from HDF5 ...")
        E = _extract_hdf5(meta, cfg, seed)
    elif cfg["paths"].get("image_dir") and os.path.exists(cfg["paths"]["image_dir"]):
        print(f">> extracting embeddings with {cfg['embedding']['model']} "
              f"(use_dino={cfg['embedding']['use_dino']}) from image_dir ...")
        paths = _resolve_image_paths(meta, cfg["paths"]["image_dir"])
        E = _extract_image_paths(meta, cfg, paths)
    else:
        dim = cfg["embedding"]["synthetic_dim"]
        print(f">> no HDF5/image_dir -> SYNTHETIC embeddings (dim={dim}).")
        E = _synthetic_embeddings(meta, dim, seed)

    np.save(ap["embeddings"], E)
    with open(ap["emb_index"], "w") as f:
        json.dump({"isic_id": isic_ids, "dim": int(E.shape[1])}, f)
    print(f">> wrote {ap['embeddings']} shape={E.shape}\n")
    return E
