"""
src/rank_models.py
==================
All ranking models compared in FABLE-5. Each returns a per-lesion score (higher =
review earlier); ranking is done within patient by the evaluator.

Rank-aware trainers (pairwise / listwise / lambda) are pure numpy so the research
engine runs fast on CPU with no torch dependency. Everything learned is fit on
TRAIN patients only; hyper-parameters / the best method are chosen on VALIDATION.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.evaluator import rank_within_patient, budget_metrics, selection_key

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Feature standardisation (impute + scale; FIT ON TRAIN ONLY)
# --------------------------------------------------------------------------- #
def standardize(feat_df, feature_cols, train_mask):
    X = feat_df[feature_cols].to_numpy(dtype=float)
    imp = SimpleImputer(strategy="median").fit(X[train_mask])
    Xi = imp.transform(X)
    sc = StandardScaler().fit(Xi[train_mask])
    return sc.transform(Xi)


def _rank_norm(train_ref):
    ref = np.sort(np.asarray(train_ref, float))
    return lambda x: np.searchsorted(ref, np.asarray(x, float), side="right") / max(len(ref), 1)


# --------------------------------------------------------------------------- #
# First-stage embedding classifier (fit on TRAIN)
# --------------------------------------------------------------------------- #
def fit_embedding_classifier(E, y, train_mask, seed=0):
    X = np.asarray(E, dtype=float)
    sc = StandardScaler().fit(X[train_mask])
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs[train_mask], y[train_mask])
    return clf.predict_proba(Xs)[:, 1]


# --------------------------------------------------------------------------- #
# Hard-negative mining for pairwise training
# --------------------------------------------------------------------------- #
def mine_pairs(F, y, patient_ids, clf_score, ctx_score, sites, E, train_mask, cfg):
    """Within each train patient, pair each malignant lesion with benign lesions,
    preferring HARD negatives: high classifier score, high context outlier, same
    anatomical site, and close in embedding space."""
    rng = np.random.default_rng(cfg.get("_seed", 0))
    cap = cfg["pairwise"]["max_negatives_per_positive"]
    hard_frac = cfg["pairwise"]["hard_negative_frac"]
    tr = np.where(train_mask)[0]
    by_pat = {}
    for i in tr:
        by_pat.setdefault(patient_ids[i], []).append(i)

    clf_n = _rank_norm(clf_score[train_mask])(clf_score)
    ctx_n = _rank_norm(ctx_score[train_mask])(ctx_score)
    pos_rows, neg_rows = [], []
    for pid, idxs in by_pat.items():
        idxs = np.array(idxs); yy = y[idxs]
        mal, ben = idxs[yy == 1], idxs[yy == 0]
        if len(mal) == 0 or len(ben) == 0:
            continue
        for m in mal:
            # hardness: classifier + context + same-site + embedding closeness
            same_site = (sites[ben] == sites[m]).astype(float) if sites is not None else 0.0
            emb_close = -np.linalg.norm(np.asarray(E[ben], float) - np.asarray(E[m], float), axis=1)
            emb_close = (emb_close - emb_close.min()) / (np.ptp(emb_close) + EPS)
            hardness = clf_n[ben] + ctx_n[ben] + same_site + emb_close
            n_take = min(cap, len(ben))
            n_hard = int(round(hard_frac * n_take))
            hard = ben[np.argsort(-hardness)[:n_hard]]
            rest = np.setdiff1d(ben, hard)
            rand = (rng.choice(rest, min(n_take - n_hard, len(rest)), replace=False)
                    if len(rest) else np.array([], dtype=int))
            chosen = np.concatenate([hard, rand]) if len(rand) else hard
            for b in chosen:
                pos_rows.append(m); neg_rows.append(b)
    return np.array(pos_rows), np.array(neg_rows)


# --------------------------------------------------------------------------- #
# Pairwise logistic (linear), optionally lambda-weighted
# --------------------------------------------------------------------------- #
def train_pairwise_logreg(F, pos, neg, masks, cfg, sample_weight=None):
    """Linear RankNet: logistic on pair differences. C chosen on validation."""
    if len(pos) == 0:
        return None
    D = F[pos] - F[neg]
    X = np.vstack([D, -D])
    yb = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    w_pairs = None if sample_weight is None else np.concatenate([sample_weight, sample_weight])
    best, best_key = None, (-np.inf, np.inf, -np.inf)
    for C in cfg["pairwise"]["C_grid"]:
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
        lr.fit(X, yb, sample_weight=w_pairs)
        w = lr.coef_[0]
        val_m, _ = _eval_scores(F @ w, masks["val_df"], cfg)
        key = selection_key(val_m, cfg)
        if key > best_key:
            best_key, best = key, w
    return best


def lambda_weights(pos, neg, clf_score, patient_ids):
    """LambdaRank-style pair weights: emphasise pairs the classifier gets wrong,
    weighting by an NDCG-like gain gap from the classifier_only ordering."""
    w = np.ones(len(pos))
    # per-patient classifier rank (1 = highest score)
    order = {}
    for pid in np.unique(patient_ids):
        m = patient_ids == pid
        idx = np.where(m)[0]
        ranks = (-clf_score[idx]).argsort().argsort() + 1
        for j, i in enumerate(idx):
            order[i] = ranks[j]
    for t, (p, n) in enumerate(zip(pos, neg)):
        gp = 1.0 / np.log2(1 + order.get(p, 1))
        gn = 1.0 / np.log2(1 + order.get(n, 1))
        w[t] = abs(gp - gn) + 0.1     # floor so every pair still contributes
    return w


# --------------------------------------------------------------------------- #
# Pairwise MLP (numpy siamese scorer, pairwise logistic loss)
# --------------------------------------------------------------------------- #
class PairwiseMLP:
    def __init__(self, d, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1))
        self.b2 = 0.0

    def _fwd(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        s = (a1 @ self.W2).ravel() + self.b2
        return s, a1, z1

    def score(self, X):
        return self._fwd(np.asarray(X, float))[0]

    def fit(self, F, pos, neg, epochs=40, lr=0.05, l2=1e-4, batch=512, seed=0):
        rng = np.random.default_rng(seed)
        n = len(pos)
        for _ in range(epochs):
            perm = rng.permutation(n)
            for st in range(0, n, batch):
                b = perm[st:st + batch]
                Xp, Xn = F[pos[b]], F[neg[b]]
                sp, ap, _ = self._fwd(Xp)
                sn, an, _ = self._fwd(Xn)
                # loss = softplus(-(sp - sn)); dL/d(sp-sn) = -sigmoid(-(sp-sn))
                g = -_sigmoid(-(sp - sn))                      # (m,)
                gW2 = (ap.T @ g[:, None] - an.T @ g[:, None]) / len(b) + l2 * self.W2
                gb2 = g.mean() - g.mean()                      # cancels (paired)
                da_p = (g[:, None] * self.W2.ravel()) * (ap > 0)
                da_n = (-g[:, None] * self.W2.ravel()) * (an > 0)
                gW1 = (Xp.T @ da_p + Xn.T @ da_n) / len(b) + l2 * self.W1
                gb1 = (da_p + da_n).mean(axis=0)
                self.W2 -= lr * gW2; self.b2 -= lr * gb2
                self.W1 -= lr * gW1; self.b1 -= lr * gb1
        return self


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------- #
# Listwise softmax (ListNet top-1, multiple positives), linear
# --------------------------------------------------------------------------- #
def train_listwise(F, y, patient_ids, train_mask, cfg):
    tr = np.where(train_mask)[0]
    groups = {}
    for i in tr:
        groups.setdefault(patient_ids[i], []).append(i)
    groups = {p: np.array(ix) for p, ix in groups.items()
              if y[np.array(ix)].sum() > 0 and len(ix) > 1}
    if not groups:
        return None
    w = np.zeros(F.shape[1])
    lr, l2, epochs = cfg["listwise"]["lr"], cfg["listwise"]["l2"], cfg["listwise"]["epochs"]
    for _ in range(epochs):
        grad = np.zeros_like(w)
        for _, ix in groups.items():
            s = F[ix] @ w; s -= s.max()
            e = np.exp(s); soft = e / e.sum()
            tgt = (y[ix] == 1).astype(float); tgt /= tgt.sum()
            grad += F[ix].T @ (soft - tgt)
        w -= lr * (grad / len(groups) + l2 * w)
    return w


# --------------------------------------------------------------------------- #
# small eval helper used during model selection
# --------------------------------------------------------------------------- #
def _eval_scores(scores_all, val_df, cfg):
    ranked = rank_within_patient(val_df, scores_all[val_df["_row"].values])
    return budget_metrics(ranked, cfg["eval"]["topk_values"], cfg["eval"]["top_pct"]), ranked


# --------------------------------------------------------------------------- #
# Orchestrate all models -> {method: per-lesion score over ALL rows}
# --------------------------------------------------------------------------- #
def build_all_scores(meta, F, group_idx, ctx_raw, clf_score, sites, E, masks, cfg):
    y = meta["malignant"].to_numpy()
    patient_ids = meta["patient_id"].to_numpy()
    train_mask, val_df = masks["train"], masks["val_df"]
    cfg = {**cfg, "_seed": cfg.get("_seed", 0)}
    want = cfg["models"]
    scores, info = {}, {}

    # random
    if "random" in want:
        scores["random"] = np.random.default_rng(cfg["_seed"]).random(len(meta))
    # classifier_only
    if "classifier_only" in want:
        scores["classifier_only"] = clf_score
    # metadata_model: logreg on group-C features only
    if "metadata_model" in want and group_idx.get("C"):
        Xc = F[:, group_idx["C"]]
        if len(np.unique(y[train_mask])) >= 2:
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Xc[train_mask], y[train_mask])
            scores["metadata_model"] = lr.predict_proba(Xc)[:, 1]
    # context_only: unsupervised centroid deviation
    if "context_only" in want:
        scores["context_only"] = ctx_raw["centroid_eucl_dist"]
    # manual_fusion_validation_selected: grid clf-weight x context feature
    if "manual_fusion_validation_selected" in want:
        clf_n = _rank_norm(clf_score[train_mask])(clf_score)
        best, best_key = None, (-np.inf, np.inf, -np.inf)
        for fname in ["centroid_eucl_dist", "knn_mean_dist", "cosine_centroid_dist"]:
            if fname not in ctx_raw:
                continue
            ctx_n = _rank_norm(ctx_raw[fname][train_mask])(ctx_raw[fname])
            for w in np.round(np.arange(0, 1.0001, 0.05), 4):
                s = w * clf_n + (1 - w) * ctx_n
                vm, _ = _eval_scores(s, val_df, cfg)
                key = selection_key(vm, cfg)
                if key > best_key:
                    best_key, best = key, (fname, float(w), s)
        if best:
            info["manual_fusion_selected"] = {"context_feature": best[0], "classifier_weight": best[1]}
            scores["manual_fusion_validation_selected"] = best[2]
    # pointwise_logreg_fusion: lesion-level BCE on all features (the Exp-2 failure mode)
    if "pointwise_logreg_fusion" in want and len(np.unique(y[train_mask])) >= 2:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(F[train_mask], y[train_mask])
        scores["pointwise_logreg_fusion"] = lr.predict_proba(F)[:, 1]

    # ---- rank-aware family: shared mined pairs ---------------------------
    need_pairs = any(m in want for m in
                     ["pairwise_rank_logreg", "pairwise_rank_mlp", "lambda_pairwise_logreg"])
    pos = neg = None
    if need_pairs:
        pos, neg = mine_pairs(F, y, patient_ids, clf_score,
                              ctx_raw["centroid_eucl_dist"], sites, E, train_mask, cfg)
        info["n_pairs"] = int(len(pos))

    if "pairwise_rank_logreg" in want and pos is not None and len(pos):
        w = train_pairwise_logreg(F, pos, neg, masks, cfg)
        if w is not None:
            scores["pairwise_rank_logreg"] = F @ w
    if "lambda_pairwise_logreg" in want and pos is not None and len(pos):
        lw = lambda_weights(pos, neg, clf_score, patient_ids)
        w = train_pairwise_logreg(F, pos, neg, masks, cfg, sample_weight=lw)
        if w is not None:
            scores["lambda_pairwise_logreg"] = F @ w
    if "pairwise_rank_mlp" in want and pos is not None and len(pos):
        mlp = PairwiseMLP(F.shape[1], hidden=cfg["pairwise"]["mlp_hidden"], seed=cfg["_seed"])
        mlp.fit(F, pos, neg, epochs=cfg["pairwise"]["mlp_epochs"],
                lr=cfg["pairwise"]["mlp_lr"], l2=cfg["pairwise"]["mlp_l2"], seed=cfg["_seed"])
        scores["pairwise_rank_mlp"] = mlp.score(F)

    # listwise
    if "listwise_softmax_ranker" in want:
        w = train_listwise(F, y, patient_ids, train_mask, cfg)
        if w is not None:
            scores["listwise_softmax_ranker"] = F @ w

    # preserve requested order
    scores = {m: scores[m] for m in want if m in scores}
    return scores, info
