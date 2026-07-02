#!/usr/bin/env python3
"""
run_04_train_eval.py
====================
Train and evaluate all models for ONE seed on held-out TEST patients, with the
paired patient-level bootstrap against classifier_only. Writes per-seed outputs.

The first-stage classifier and the classifier-derived features (groups A + D) are
computed here, per seed, fit on TRAIN patients only — so the reusable feature
table (groups B + C) stays leakage-free.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.io_utils import read_table, write_table

from src.config import load_config, artifact_paths
from src import feature_builder as fb
from src import rank_models as rm
from src import evaluator as ev
from src import bootstrap as bs
from src import report_writer as rw

ID_COLS = ["isic_id", "patient_id", "malignant"]


def _load_seed_inputs(cfg, seed):
    ap = artifact_paths(cfg)
    meta = read_table(ap["metadata"]).reset_index(drop=True)
    feats = read_table(ap["features"]).reset_index(drop=True)
    E = np.load(ap["embeddings"], mmap_mode="r")
    with open(os.path.join(ap["splits_dir"], f"split_seed_{seed}.json")) as f:
        split = json.load(f)
    with open(os.path.join(ap["work_dir"], fb.FEATURE_GROUPS_FILE)) as f:
        groups = json.load(f)
    return meta, feats, E, split, groups


def train_eval_seed(cfg, seed, out_dir=None):
    cfg = {**cfg, "_seed": seed}
    ap = artifact_paths(cfg)
    out_dir = out_dir or os.path.join(cfg["paths"]["results_dir"], f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    meta, feats, E, split, groups = _load_seed_inputs(cfg, seed)
    meta = meta.copy(); meta["_row"] = np.arange(len(meta))
    y = meta["malignant"].to_numpy()
    pid = meta["patient_id"].to_numpy()

    train_mask = np.isin(pid, split["train"])
    val_mask = np.isin(pid, split["val"])
    test_mask = np.isin(pid, split["test"])

    # first-stage classifier (fit on TRAIN embeddings) -> groups A + D
    clf_score = rm.fit_embedding_classifier(E, y, train_mask, seed=seed)
    ad = fb.classifier_and_disagreement_features(meta, clf_score, feats)

    # assemble full feature frame: B + C (static) + A + D (per seed)
    static_cols = groups["B_context"] + groups["C_metadata"]
    full = pd.concat([feats[static_cols].reset_index(drop=True),
                      ad.reset_index(drop=True)], axis=1)
    feature_cols = list(full.columns)
    F = rm.standardize(full, feature_cols, train_mask)

    group_idx = {"C": [feature_cols.index(c) for c in groups["C_metadata"] if c in feature_cols]}
    ctx_raw = {c: feats[c].to_numpy() for c in
               ["centroid_eucl_dist", "knn_mean_dist", "cosine_centroid_dist"]
               if c in feats.columns}
    sites = (meta["anatom_site_general"].astype(str).to_numpy()
             if "anatom_site_general" in meta.columns else None)

    masks = {"train": train_mask,
             "val_df": meta[val_mask].copy(), "test_df": meta[test_mask].copy()}

    scores, info = rm.build_all_scores(meta, F, group_idx, ctx_raw, clf_score,
                                       sites, np.asarray(E), masks, cfg)

    # evaluate on TEST; collect summary + per-patient tables + predictions ---
    test_df = meta[test_mask].copy()
    summary_rows, per_patient, best_key, best_method = [], {}, (-np.inf,), None
    ranks_table = test_df[["patient_id", "malignant"]].copy()
    ranks_table = ranks_table.rename(columns={"malignant": "true_label"})
    ranks_table["lesion_id"] = meta.loc[test_mask, "isic_id"].values
    ranks_table["classifier_score"] = clf_score[test_mask]
    ranks_table["context_score"] = ctx_raw["centroid_eucl_dist"][test_mask]

    val_keys = {}
    for method, s in scores.items():
        m_test, ranked_test = ev.evaluate(test_df, s[test_mask], cfg)
        m_val, _ = ev.evaluate(masks["val_df"], s[val_mask], cfg)
        val_keys[method] = ev.selection_key(m_val, cfg)
        row = {"method": method, "split": "test", **m_test}
        summary_rows.append(row)
        per_patient[method] = ev.per_patient_first_rank(ranked_test)
        ranks_table[f"rank_{method}"] = ranked_test.sort_values("_row")[ev.RANK].values \
            if "_row" in ranked_test else ranked_test[ev.RANK].values

    summary_df = pd.DataFrame(summary_rows)
    best_method = max(val_keys, key=val_keys.get)

    # paired patient bootstrap vs classifier_only --------------------------
    merged = bs.build_merged(per_patient)
    metrics = bs.bootstrap_metrics_list(cfg["eval"]["topk_values"])
    paired_records = []
    paired_rows = {}
    for method in scores:
        if method == "classifier_only":
            continue
        rep = bs.paired_bootstrap(merged, "fr_classifier_only", f"fr_{method}",
                                  metrics, n_bootstrap=cfg["eval"]["n_bootstrap"], seed=seed)
        paired_rows[method] = rep
        for r in rep:
            paired_records.append({"method": method, **r})

    # outputs --------------------------------------------------------------
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.DataFrame(paired_records).to_csv(
        os.path.join(out_dir, "paired_comparison_report.csv"), index=False)

    best_ranked = ev.rank_within_patient(test_df, scores[best_method][test_mask])
    preds = pd.DataFrame({
        "patient_id": best_ranked["patient_id"].values,
        "lesion_id": meta.loc[test_mask, "isic_id"].values,
        "true_label": best_ranked["malignant"].values,
        "classifier_score": clf_score[test_mask],
        "context_score": ctx_raw["centroid_eucl_dist"][test_mask],
        "fused_score": best_ranked["_score"].values,
        "method": best_method,
        "patient_rank": best_ranked[ev.RANK].values,
    }).sort_values(["patient_id", "patient_rank"])
    preds.to_csv(os.path.join(out_dir, "predictions_best.csv"), index=False)
    write_table(ranks_table, os.path.join(out_dir, "all_method_ranks.parquet"), index=False)
    with open(os.path.join(out_dir, "seed_meta.json"), "w") as f:
        json.dump({"seed": seed, "best_method": best_method, "info": info}, f, default=str)
    rw.per_seed_report(os.path.join(out_dir, "report_seed.md"),
                       seed, summary_df, paired_rows, best_method, info, cfg)

    print("=" * 70)
    print(f"SEED {seed} — TEST SUMMARY  (best on val: {best_method})")
    print("=" * 70)
    show = ["method"] + [f"recall@{k}" for k in cfg["eval"]["topk_values"]] + [
        "mean_rank_first_malignant", "normalized_percentile_rank", "auroc"]
    print(summary_df[show].round(3).to_string(index=False))
    print(f">> outputs -> {out_dir}/\n")
    return summary_df, paired_records, best_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fable5_smoke.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_mode", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    a = ap.parse_args()
    ov = {"run_mode": a.run_mode} if a.run_mode else {}
    if a.work_dir: ov.setdefault("paths", {})["work_dir"] = a.work_dir
    if a.results_dir: ov.setdefault("paths", {})["results_dir"] = a.results_dir
    cfg = load_config(a.config, ov)
    train_eval_seed(cfg, a.seed)


if __name__ == "__main__":
    main()
