"""
run_experiment0.py  (v2 — post Gemini audit)
--------------------------------------------
Full pipeline with all audit fixes applied:
  - Score-flip sanity check for all rerankers
  - Fixed NNT (method-dependent, aggregate sensitivity)
  - Method-dependent candidate reduction at fixed sensitivity
  - Bootstrap 95% CI for SE@k, MRR, NNT, AUROC
  - Separate positive-patient / all-patient reporting
  - Score distribution plots
  - CARD ablation table
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path="configs/experiment0.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(cfg_path="configs/experiment0.yaml", csv_override=None):
    cfg    = load_config(cfg_path)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(cfg["experiment"]["device"])
    out    = Path(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "experiment0_config.yaml")

    print("\n" + "█" * 70)
    print(" EXPERIMENT 0 v2 — Second-Stage Contextual Reranking")
    print("█" * 70)

    # ── Step 0: Data ──────────────────────────────────────────────────────────
    mode = "real" if csv_override else cfg["data"]["mode"]
    if csv_override:
        cfg["data"]["csv_path"] = csv_override

    if mode == "toy":
        print("\n[Step 0] Generating toy dataset...")
        from src.data.toy_generator import generate_toy_dataset
        tc = cfg["toy"]
        df, raw_embeddings = generate_toy_dataset(
            n_patients=tc["n_patients"],
            lesions_per_patient_min=tc["lesions_per_patient_min"],
            lesions_per_patient_max=tc["lesions_per_patient_max"],
            malignant_rate=tc["malignant_rate"],
            malignant_per_patient=tc["malignant_per_patient"],
            embedding_dim=tc["embedding_dim"],
            patient_spread=tc["patient_spread"],
            benign_noise=tc["benign_noise"],
            malignant_shift=tc["malignant_shift"],
            malignant_noise=tc["malignant_noise"],
            seed=cfg["experiment"]["seed"],
            output_csv=tc["output_csv"],
            output_embeddings=tc["output_embeddings"],
        )
        embedding_dim = tc["embedding_dim"]
    else:
        print(f"\n[Step 0] Loading: {cfg['data']['csv_path']}")
        df = pd.read_csv(cfg["data"]["csv_path"])
        emb_path = cfg["classifier"]["embeddings_path"].replace(".npz", "_raw.npy")
        raw_embeddings = np.load(emb_path)
        embedding_dim  = raw_embeddings.shape[1]

    # ── Step 1: Audit ─────────────────────────────────────────────────────────
    print("\n[Step 1] Data audit...")
    from src.data.audit import audit_dataset
    audit_dataset(df, report_path=str(out / "data_audit.txt"))

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    print("\n[Step 2] Patient-disjoint splitting...")
    from src.data.splitter import split_by_patient, verify_splits
    sp = cfg["split"]
    df = split_by_patient(df, sp["train_frac"], sp["val_frac"],
                          sp["test_frac"], cfg["experiment"]["seed"],
                          sp["split_col"], sp["report_path"])
    verify_splits(df)
    if mode == "toy":
        df.to_csv(cfg["toy"]["output_csv"], index=False)

    # ── Step 3: Train classifier ──────────────────────────────────────────────
    print("\n[Step 3] Training independent classifier...")
    from src.data.dataset import LesionEmbeddingDataset
    from src.models.classifier import (ClassifierTrainer, extract_embeddings,
                                        save_embeddings)

    train_ds = LesionEmbeddingDataset(df, raw_embeddings, split="train")
    val_ds   = LesionEmbeddingDataset(df, raw_embeddings, split="val")

    trainer = ClassifierTrainer(cfg, embedding_dim=embedding_dim)
    trainer.fit(train_ds, val_ds, cfg["classifier"]["checkpoint_path"])

    # ── Step 4: Extract embeddings ────────────────────────────────────────────
    print("\n[Step 4] Extracting embeddings for all splits...")
    all_ds = LesionEmbeddingDataset(df, raw_embeddings, split=None)
    ext = extract_embeddings(trainer.model, all_ds, device)
    save_embeddings(ext, cfg["classifier"]["embeddings_path"])

    # ── Step 5: Calibration ───────────────────────────────────────────────────
    print("\n[Step 5] Calibrating on val patients only...")
    from src.models.calibration import calibrate

    val_lids = set(df[df["split"] == "val"]["lesion_id"].tolist())
    val_mask = np.array([lid in val_lids for lid in ext["lesion_ids"]])

    cal_probs_all, cal_report = calibrate(
        raw_logits=None,
        val_logits=ext["logits"][val_mask],
        val_labels=ext["labels"][val_mask],
        all_logits=ext["logits"],
        all_labels=ext["labels"],
        method=cfg["calibration"]["method"],
        report_path=cfg["calibration"]["report_path"],
    )
    np.save(str(out / "calibrated_probs.npy"), cal_probs_all)

    lid2calprob = {lid: float(cal_probs_all[i])
                   for i, lid in enumerate(ext["lesion_ids"])}
    cls_embs    = ext["embeddings"]

    # ── Step 6: Build patient groups ──────────────────────────────────────────
    print("\n[Step 6] Building patient groups...")
    df_idx = df.reset_index(drop=True)
    lid2cemb_idx = {lid: i for i, lid in enumerate(ext["lesion_ids"])}

    def build_groups(split_name):
        sub = df_idx[df_idx["split"] == split_name]
        groups = []
        for pid, grp in sub.groupby("patient_id"):
            idxs = [lid2cemb_idx[lid] for lid in grp["lesion_id"]]
            groups.append({
                "patient_id": pid,
                "embeddings": torch.from_numpy(cls_embs[idxs]).float(),
                "labels": torch.tensor(grp["malignant"].values, dtype=torch.long),
                "lesion_ids": grp["lesion_id"].tolist(),
                "n_lesions": len(grp),
            })
        return groups

    g_train = build_groups("train")
    g_val   = build_groups("val")
    g_test  = build_groups("test")
    emb_dim_cls = cls_embs.shape[1]

    print(f"  train={len(g_train)} val={len(g_val)} test={len(g_test)} patients")

    # ── Step 7: Train learnable rerankers ─────────────────────────────────────
    print("\n[Step 7] Training learnable rerankers (with score-flip check)...")
    from src.training.trainer import (
        train_toar_lite, train_set_transformer, score_with_flip
    )
    from src.rerankers.card import train_all_card_variants

    toar_lite      = train_toar_lite(g_train, g_val, emb_dim_cls, cfg, device)
    set_transformer= train_set_transformer(g_train, g_val, emb_dim_cls, cfg, device)

    card_models = train_all_card_variants(
        patient_groups_train=g_train,
        patient_groups_val=g_val,
        cal_probs=cal_probs_all,
        all_lesion_ids=ext["lesion_ids"],
        cfg=cfg,
    )

    # ── Step 8: Score-flip check for non-trainable rerankers ─────────────────
    print("\n[Step 8] Scoring all rerankers on test (with flip check)...")
    from src.rerankers.rerankers import (
        AbsoluteRiskReranker, CentroidReranker, KNNReranker,
        apply_score_flip_if_needed,
    )

    # Build val flat arrays for non-trainable flip check
    val_flat_labels = np.concatenate([
        grp["labels"].numpy() for grp in g_val
    ])

    def score_patient_set(reranker, groups, use_flip_check=False,
                          val_scores_for_check=None):
        """Score all patients; return flat array aligned with groups."""
        all_sc = []
        for grp in groups:
            emb  = grp["embeddings"].numpy()
            lids = grp["lesion_ids"]
            abs_p = np.array([lid2calprob.get(lid, 0.5) for lid in lids])
            sc = reranker.score(embeddings=emb, abs_probs=abs_p)
            all_sc.append(sc)
        return np.concatenate(all_sc)

    def score_with_val_flip(reranker, g_val_grps, g_test_grps, name):
        """Score val first to detect flip, then apply to test."""
        val_sc = score_patient_set(reranker, g_val_grps)
        val_lb = np.concatenate([grp["labels"].numpy() for grp in g_val_grps])
        _, flipped, auroc = apply_score_flip_if_needed(
            val_sc, val_lb, name=name, verbose=True
        )
        test_sc = score_patient_set(reranker, g_test_grps)
        if flipped:
            test_sc = -test_sc
        return test_sc

    non_trainable = {
        "absolute_risk":      AbsoluteRiskReranker(),
        "centroid_euclidean": CentroidReranker("euclidean"),
        "centroid_cosine":    CentroidReranker("cosine"),
        "knn_k3":             KNNReranker(k=3),
        "knn_k5":             KNNReranker(k=5),
    }

    scores_test = {}
    for name, ranker in non_trainable.items():
        scores_test[name] = score_with_val_flip(ranker, g_val, g_test, name)

    # Trainable rerankers use the flip flag set during training
    for name, model in [("toar_lite", toar_lite),
                         ("set_transformer", set_transformer)]:
        all_sc = []
        for grp in g_test:
            emb  = grp["embeddings"].numpy()
            lids = grp["lesion_ids"]
            abs_p = np.array([lid2calprob.get(lid, 0.5) for lid in lids])
            sc = score_with_flip(model, emb, abs_p)
            all_sc.append(sc)
        scores_test[name] = np.concatenate(all_sc)
        print(f"  Scored (trainable): {name}")

    for variant, model in card_models.items():
        all_sc = []
        for grp in g_test:
            emb  = grp["embeddings"].numpy()
            lids = grp["lesion_ids"]
            abs_p = np.array([lid2calprob.get(lid, 0.5) for lid in lids])
            sc = model.score(embeddings=emb, abs_probs=abs_p)
            all_sc.append(sc)
        scores_test[f"card_{variant}"] = np.concatenate(all_sc)
        print(f"  Scored (CARD):     card_{variant}")

    # ── Step 9: Metrics ───────────────────────────────────────────────────────
    print("\n[Step 9] Computing metrics with bootstrap CIs...")
    from src.metrics.metrics import (
        compute_all_metrics, save_metrics, print_metrics_table,
        print_card_ablation_table, build_predictions_csv,
    )

    k_vals = cfg["metrics"]["k_values"]
    s_tgts = cfg["metrics"]["sensitivity_targets"]

    results = compute_all_metrics(
        patient_groups=g_test,
        scores_dict=scores_test,
        k_values=k_vals,
        sensitivity_targets=s_tgts,
        n_boot=500,
    )

    print_metrics_table(results, k_vals)
    print_card_ablation_table(results)
    save_metrics(results, cfg["metrics"]["results_path"])

    # ── Step 10: Outputs ──────────────────────────────────────────────────────
    print("\n[Step 10] Saving predictions and plots...")
    build_predictions_csv(g_test, scores_test,
                          cfg["metrics"]["predictions_path"])

    from src.metrics.plots import (
        plot_se_at_k, plot_precision_at_k, plot_calibration_curve,
        plot_nnt_comparison, plot_score_distributions,
        plot_card_ablation_table,
    )

    plots = cfg["metrics"]["plots_dir"]
    plot_se_at_k(results, k_vals, f"{plots}/se_at_k.png")
    plot_precision_at_k(results, k_vals, f"{plots}/precision_at_k.png")
    raw_probs = 1 / (1 + np.exp(-ext["logits"]))

    # Calibration plot uses test-split lesions only
    test_lids = set(df[df["split"] == "test"]["lesion_id"].tolist())
    test_mask = np.array([lid in test_lids for lid in ext["lesion_ids"]])
    plot_calibration_curve(
        raw_probs[test_mask], cal_probs_all[test_mask],
        ext["labels"][test_mask], f"{plots}/calibration_curve.png"
    )
    plot_nnt_comparison(results, f"{plots}/nnt_comparison.png", s_tgts)
    plot_score_distributions(g_test, scores_test, plots)
    plot_card_ablation_table(results, f"{plots}/card_ablation_table.png",
                              k_values=k_vals)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print(" EXPERIMENT 0 v2 COMPLETE")
    print("█" * 70)
    print(f"\n  Outputs in: {out}/")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out)}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n[Interpretation]")
    print("\n  ⚠️  DISCLAIMER: Toy results are only pipeline validation,")
    print("      not evidence of clinical performance.")
    print("      All numbers below are from synthetic data.")
    print("      Repeat with real SLICE-3D data before drawing conclusions.\n")

    abs_se5  = results.get("absolute_risk", {}).get("SE@5")
    best_name = max(results, key=lambda k: results[k].get("SE@5") or 0)
    best_se5 = results[best_name].get("SE@5")

    if abs_se5 is not None and best_se5 is not None:
        gain = best_se5 - abs_se5
        print(f"  Absolute-risk SE@5      : {abs_se5:.3f}")
        print(f"  Best reranker SE@5      : {best_se5:.3f}  ({best_name})")
        if gain < 0.02:
            print("\n  ⚠️  No clear gain from contextual reranking on toy data.")
            print("      This may be due to small test set (few positive patients).")
            print("      Expected on toy: NNT@80 == NNT@90 when < ~10 positive patients.")
        else:
            print(f"\n  ✅  Contextual reranking gains {gain:.3f} SE@5.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  default="configs/experiment0.yaml")
    parser.add_argument("--csv",  default=None)
    args = parser.parse_args()
    main(cfg_path=args.cfg, csv_override=args.csv)
