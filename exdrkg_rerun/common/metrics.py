"""Evaluation metrics for the reviewer rerun (one definition for every model).

AUROC / AUPRC : pooled over all test pairs (sklearn roc_auc_score, average_precision_score).
Hit@k / MRR   : for each test disease with >=1 positive, rank the candidate drugs that appear
                with that disease in the test file by score. Hit@k = fraction of positive pairs
                ranked in the top k; MRR = mean reciprocal rank of positive pairs.
CIs           : bootstrap over pairs, stratified by label (disease-family labels are not available
                for DOID ids in this repository), 1,000 resamples, seed 42.
Significance  : DeLong test for AUROC; paired bootstrap for AUPRC / Hit@10 differences.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

def ranking_metrics(df: pd.DataFrame, score: np.ndarray, ks=(1, 5, 10, 20)):
    d = df[["disease_id", "label"]].copy(); d["s"] = np.asarray(score)
    d["rank"] = d.groupby("disease_id")["s"].rank(ascending=False, method="average")  # ties share the average rank
    d = d[(d.label == 1)]
    ranks = d["rank"].values
    out = {f"Hit@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks).mean())
    return out

def compute_metrics(df, score):
    y = df.label.values
    m = {"AUROC": float(roc_auc_score(y, score)), "AUPRC": float(average_precision_score(y, score)),
         "AUPRC_random": float(y.mean())}
    m.update(ranking_metrics(df, score))
    return m

def _strata(df, strata=None):
    """Bootstrap strata: (disease family x label) when a family column is given, else label only."""
    lab = df.label.values.astype(str)
    return lab if strata is None else np.char.add(np.asarray(strata).astype(str), lab)

def _resample_idx(groups, rng):
    idx = []
    for g in np.unique(groups):
        members = np.where(groups == g)[0]; idx.append(rng.choice(members, len(members)))
    return np.concatenate(idx)

def bootstrap_ci(df, score, n=1000, seed=42, keys=("AUROC", "AUPRC", "Hit@10", "MRR"), strata=None):
    rng = np.random.default_rng(seed); groups = _strata(df, strata); res = {k: [] for k in keys}
    for _ in range(n):
        idx = _resample_idx(groups, rng)
        sub = df.iloc[idx].reset_index(drop=True); m = compute_metrics(sub, np.asarray(score)[idx])
        for k in keys: res[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}

def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong_test(y, s1, s2):
    """Two-sided DeLong test for correlated AUROCs (Sun & Xu 2014). Returns (auc1, auc2, p)."""
    y = np.asarray(y); order = np.argsort(-y); m = int(y.sum()); n = len(y) - m
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    aucs = np.zeros(2); v01 = np.zeros((2, m)); v10 = np.zeros((2, n))
    for r in range(2):
        pos, neg = preds[r, :m], preds[r, m:]
        tx, ty, tz = _compute_midrank(pos), _compute_midrank(neg), _compute_midrank(preds[r])
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01[r] = (tz[:m] - tx) / n; v10[r] = 1 - (tz[m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0: return aucs[0], aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))

def paired_bootstrap_p(df, s1, s2, metric, n=1000, seed=42, strata=None):
    """Two-sided paired-bootstrap p-value for metric(s1) - metric(s2) = 0, resampling the SAME pairs for both models."""
    rng = np.random.default_rng(seed); groups = _strata(df, strata); diffs = []
    s1, s2 = np.asarray(s1), np.asarray(s2)
    for _ in range(n):
        idx = _resample_idx(groups, rng); sub = df.iloc[idx].reset_index(drop=True)
        diffs.append(compute_metrics(sub, s1[idx])[metric] - compute_metrics(sub, s2[idx])[metric])
    diffs = np.array(diffs)
    return float(min(1.0, 2 * (min((diffs <= 0).sum(), (diffs >= 0).sum()) + 1) / (n + 1)))
