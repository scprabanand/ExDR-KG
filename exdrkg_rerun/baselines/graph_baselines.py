"""Real, CPU-only baselines on the leakage-safe HKG (no LLM, no GPU, no simulated numbers).

Models
  random               uniform random scores (seed 42)
  shortest_path        score = -shortest-path distance (no training)
  aa2 / aa3            Adamic-Adar weighted 2-hop / 3-hop walk mass (no training)
  gbm_mechanism        gradient boosting on path statistics only (indication edges never visible)
  gbm_mechanism+ind    same + collaborative features from known TRAIN indications (K-fold masked)
Outputs (results/rerun/<split>/): scores_<model>.csv, metrics.json, table_RA.csv, significance.json
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from exdrkg_rerun.baselines.graph_features import HKG
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test, paired_bootstrap_p

def pair_features(H, base, ind, df, cols):
    di = df.drug_id.map(H.did).values; si = df.disease_id.map(H.sid).values
    feats = {**base, **(ind or {})}
    return np.column_stack([feats[c][di, si] for c in cols]), di, si

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard")
    ap.add_argument("--out", default=None); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); root = Path(a.root); out = Path(a.out or root / "results" / "rerun" / a.split); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42); t0 = time.time()
    H = HKG(root / "data/processed/unified_kg.pt"); print("graph loaded", H.N, "nodes", H.A.nnz // 2, "undirected edges", flush=True)
    sp_ = {k: pd.read_csv(root / f"data/splits/{a.split}/{k}.csv") for k in ["train", "val", "test"] if (root / f"data/splits/{a.split}/{k}.csv").exists()}
    base = H.base_feature_matrices(); print("base features", round(time.time() - t0), "s", flush=True)
    base_cols = [c for c in base if c not in ("walk1",)]
    tr, va, te = sp_["train"], sp_["val"], sp_["test"]
    # ---- mechanism-only GBM
    Xtr, *_ = pair_features(H, base, None, tr, base_cols); Xva, *_ = pair_features(H, base, None, va, base_cols); Xte, *_ = pair_features(H, base, None, te, base_cols)
    def fit(X, y):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                           class_weight="balanced", early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=42)
        return m.fit(X, y)
    m1 = fit(Xtr, tr.label.values)
    # ---- + collaborative indication features, K-fold masked on train positives
    tpos = tr[tr.label == 1]; Tall = np.zeros((H.nd, H.ns), np.float32)
    for d, s in zip(tpos.drug_id.map(H.did), tpos.disease_id.map(H.sid)): Tall[d, s] = 1
    ind_all = H.indication_features(Tall)
    fold = rng.integers(0, a.folds, len(tr)); ind_cols = list(ind_all)
    Xtr2 = np.zeros((len(tr), len(base_cols) + len(ind_cols)), np.float32)
    di = tr.drug_id.map(H.did).values; si = tr.disease_id.map(H.sid).values
    for k in range(a.folds):
        Tk = Tall.copy(); hid = tr[(fold == k) & (tr.label == 1)]
        for d, s in zip(hid.drug_id.map(H.did), hid.disease_id.map(H.sid)): Tk[d, s] = 0
        indk = H.indication_features(Tk); rows = np.where(fold == k)[0]
        F = {**base, **indk}
        Xtr2[rows] = np.column_stack([F[c][di[rows], si[rows]] for c in base_cols + ind_cols])
    Xva2, *_ = pair_features(H, base, ind_all, va, base_cols + ind_cols); Xte2, *_ = pair_features(H, base, ind_all, te, base_cols + ind_cols)
    m2 = fit(Xtr2, tr.label.values); print("models trained", round(time.time() - t0), "s", flush=True)
    di_t = te.drug_id.map(H.did).values; si_t = te.disease_id.map(H.sid).values
    scores = {"random": rng.random(len(te)),
              "shortest_path": -base["dist"][di_t, si_t] + 1e-6 * rng.random(len(te)),   # tiny noise only breaks exact ties
              "aa2": base["aa2"][di_t, si_t] + 1e-9 * rng.random(len(te)),
              "aa3": base["aa3"][di_t, si_t] + 1e-9 * rng.random(len(te)),
              "gbm_mechanism": m1.predict_proba(Xte)[:, 1], "gbm_mechanism+ind": m2.predict_proba(Xte2)[:, 1]}
    val_scores = {"gbm_mechanism": m1.predict_proba(Xva)[:, 1], "gbm_mechanism+ind": m2.predict_proba(Xva2)[:, 1]}
    res = {}; rows = []
    for k, s in scores.items():
        pd.DataFrame({"drug_id": te.drug_id, "disease_id": te.disease_id, "label": te.label, "score": s}).to_csv(out / f"scores_{k}.csv", index=False)
        m = compute_metrics(te, s); ci = bootstrap_ci(te, s, n=a.boot); res[k] = {"metrics": m, "ci95": ci}
        rows.append({"Model": k, "AUROC (95% CI)": f"{m['AUROC']:.3f} ({ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f})",
                     "AUPRC (95% CI)": f"{m['AUPRC']:.3f} ({ci['AUPRC'][0]:.3f}-{ci['AUPRC'][1]:.3f})",
                     "Hit@1": round(m["Hit@1"], 3), "Hit@10": round(m["Hit@10"], 3), "MRR": round(m["MRR"], 3)})
    for k in val_scores: res[k]["val"] = compute_metrics(va, val_scores[k])
    ref = "gbm_mechanism+ind"; sig = {}
    for k in scores:
        if k == ref: continue
        a1, a2, p = delong_test(te.label.values, scores[ref], scores[k]); sig[k] = {"auc_ref": a1, "auc_other": a2, "delong_p": p,
            "auprc_boot_p": paired_bootstrap_p(te, scores[ref], scores[k], "AUPRC", n=a.boot)}
    tab = pd.DataFrame(rows); tab.to_csv(out / "table_RA.csv", index=False)
    json.dump({"n_test": len(te), "n_pos": int(te.label.sum()), "features": {"mechanism": base_cols, "indication": ind_cols},
               "results": res, "significance_vs_" + ref: sig, "runtime_s": round(time.time() - t0)}, open(out / "metrics.json", "w"), indent=1)
    print(tab.to_string(index=False)); print(json.dumps(sig, indent=1))

if __name__ == "__main__": main()
