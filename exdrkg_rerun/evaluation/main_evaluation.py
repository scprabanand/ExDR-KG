"""Step 2: one authoritative results table from per-model score files (Table R-A).

Reads results/rerun/main/scores/<model>.csv (columns drug_id, disease_id, label, score), one file per model, verifies
each file against the held-out test file, then computes everything from those files in a single run:

  * AUROC and AUPRC pooled over all test pairs (sklearn roc_auc_score / average_precision_score);
  * Hit@1, Hit@10 and MRR with ONE definition for every model: for each test disease with at least one positive
    drug, all candidate drugs paired with that disease in the test file are ranked by score (ties share the average
    rank); Hit@k is the fraction of positive pairs ranked in the top k; MRR is the mean reciprocal rank of positives;
  * 95% CIs from 1,000 bootstrap resamples (seed 42) stratified by disease family x label when --family-map is given,
    otherwise by label only (a warning is printed and recorded);
  * ExDR-KG vs each baseline: DeLong test for AUROC, paired bootstrap for AUPRC and Hit@10, Bonferroni-corrected for
    the number of baselines (0.05 / 7 = 0.0071);
  * random-chance AUPRC (= positive fraction of the test file).

Expected model files (any subset; names are the table row labels):
  Metapath2vec HAN R-GCN CompGCN TxGNN BioGPT GPT-4+RAG ExDR-KG
Score files are produced by score_exdrkg.py, gnn_baselines.py, txgnn.py, biogpt.py and
llm_api.py. Nothing in this script is typed in; every number comes from the score files.

Usage
  python -m exdrkg_rerun.evaluation.main_evaluation --root . --split standard_degree_matched \
      --scores results/rerun/main/scores --family-map data/disease_family.csv --ref ExDR-KG
Outputs (results/rerun/main/): table_RA.csv, table_RA.md, table_RA.tex, metrics.json, definitions.txt
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test, paired_bootstrap_p

ORDER = ["Metapath2vec", "HAN", "R-GCN", "CompGCN", "TxGNN", "BioGPT", "GPT-4+RAG", "ExDR-KG"]
DEFINITIONS = """Metric definitions (identical for every model)
AUROC, AUPRC: computed over all held-out test pairs pooled (sklearn roc_auc_score, average_precision_score).
Hit@k: for each test disease with at least one positive drug, all candidate drugs paired with that disease in the test
file are ranked by predicted score (ties share the average rank); Hit@k is the fraction of positive pairs ranked in the
top k. MRR: mean reciprocal rank of the positive pairs under the same ranking.
Random-chance AUPRC equals the positive fraction of the test file.
Confidence intervals: 1,000 bootstrap resamples (seed 42), stratified by {strata}.
Significance: DeLong test (AUROC); paired bootstrap on identical resamples (AUPRC, Hit@10); Bonferroni correction for
{k} comparisons (threshold {thr:.4f}).
"""


def load_scores(path: Path, test: pd.DataFrame, name: str) -> np.ndarray:
    s = pd.read_csv(path)
    need = {"drug_id", "disease_id", "label", "score"}
    if not need <= set(s.columns): raise ValueError(f"{name}: columns must include {sorted(need)}")
    if len(s) != len(test): raise ValueError(f"{name}: {len(s)} rows, expected {len(test)} (score EVERY test pair)")
    if s.score.isna().any() or not np.isfinite(s.score.values).all(): raise ValueError(f"{name}: NaN/inf scores")
    # align on the pair key, never on row order
    m = test[["drug_id", "disease_id", "label"]].merge(s, on=["drug_id", "disease_id"], how="left", suffixes=("", "_s"))
    if len(m) != len(test) or m.score.isna().any(): raise ValueError(f"{name}: pairs do not match the test file")
    if (m.label != m.label_s).any(): raise ValueError(f"{name}: labels disagree with the test file")
    return m.score.values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard")
    ap.add_argument("--scores", default="results/rerun/main/scores"); ap.add_argument("--out", default="results/rerun/main")
    ap.add_argument("--family-map", default=None, help="CSV with columns disease_id,family for stratified bootstrap")
    ap.add_argument("--ref", default="ExDR-KG"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(root / "data/splits" / a.split / "test.csv").reset_index(drop=True)
    if test.duplicated(["drug_id", "disease_id"]).any():
        raise SystemExit("test file has repeated (drug, disease) rows; run rebuild_splits.py first")
    strata = None
    if a.family_map:
        fam = pd.read_csv(root / a.family_map).set_index("disease_id").family
        strata = test.disease_id.map(fam).fillna("Other").values
    else:
        print("WARNING: no --family-map; bootstrap is stratified by label only (state this in the manuscript)")
    files = {p.stem.replace("_", " ") if False else p.stem: p for p in sorted(Path(root / a.scores).glob("*.csv"))}
    models = [m for m in ORDER if m in files] + [m for m in files if m not in ORDER]
    if a.ref not in models: raise SystemExit(f"reference model '{a.ref}' has no score file in {a.scores}")
    S = {m: load_scores(files[m], test, m) for m in models}
    baselines = [m for m in models if m != a.ref]; k = max(len(baselines), 1); thr = 0.05 / k
    res, rows = {}, []
    for m in models:
        met = compute_metrics(test, S[m]); ci = bootstrap_ci(test, S[m], n=a.boot, strata=strata)
        res[m] = {"metrics": met, "ci95": ci}
    for m in models:
        met, ci = res[m]["metrics"], res[m]["ci95"]
        row = {"Model": m,
               "AUROC (95% CI)": f"{met['AUROC']:.3f} ({ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f})",
               "AUPRC (95% CI)": f"{met['AUPRC']:.3f} ({ci['AUPRC'][0]:.3f}-{ci['AUPRC'][1]:.3f})",
               "Hit@1": f"{met['Hit@1']:.3f}", "Hit@10": f"{met['Hit@10']:.3f}", "MRR": f"{met['MRR']:.3f}"}
        if m != a.ref:
            auc_r, auc_m, p_auc = delong_test(test.label.values, S[a.ref], S[m])
            p_prc = paired_bootstrap_p(test, S[a.ref], S[m], "AUPRC", n=a.boot, strata=strata)
            p_h10 = paired_bootstrap_p(test, S[a.ref], S[m], "Hit@10", n=a.boot, strata=strata)
            res[m]["vs_ref"] = {"delong_p_AUROC": p_auc, "boot_p_AUPRC": p_prc, "boot_p_Hit@10": p_h10, "bonferroni_threshold": thr,
                                "significant_all_three": bool(max(p_auc, p_prc, p_h10) < thr),
                                "ref_minus_model": {"AUROC": auc_r - auc_m, "AUPRC": res[a.ref]["metrics"]["AUPRC"] - met["AUPRC"],
                                                    "Hit@10": res[a.ref]["metrics"]["Hit@10"] - met["Hit@10"]}}
            row["p vs " + a.ref + " (AUROC / AUPRC / Hit@10)"] = f"{p_auc:.4g} / {p_prc:.4g} / {p_h10:.4g}"
        else:
            row["p vs " + a.ref + " (AUROC / AUPRC / Hit@10)"] = "reference"
        rows.append(row)
    table = pd.DataFrame(rows)
    npos = int(test.label.sum()); rand_auprc = npos / len(test)
    flags = []
    ref_m = res[a.ref]["metrics"]
    if ref_m["AUROC"] < 0.9 and ref_m["AUPRC"] > 0.7:   # sanity check from the rerun plan
        flags.append(f"CHECK: {a.ref} has AUROC {ref_m['AUROC']:.3f} but AUPRC {ref_m['AUPRC']:.3f}; with ~1% positives this is unusual. "
                     "Confirm AUPRC is pooled over all test pairs, not per disease or on a balanced subset.")
    if any(res[m]["metrics"]["AUROC"] < 0.5 for m in models): flags.append("CHECK: a model has AUROC < 0.5 (inverted scores?).")
    table.to_csv(out / "table_RA.csv", index=False)
    (out / "table_RA.md").write_text(table.to_markdown(index=False) + f"\n\nRandom-chance AUPRC = {npos}/{len(test)} = {rand_auprc:.4f}\n" + "\n".join(flags))
    (out / "table_RA.tex").write_text(table.to_latex(index=False, escape=True))
    (out / "definitions.txt").write_text(DEFINITIONS.format(strata=("disease family x label" if strata is not None else "label only (disease-family map not available)"), k=k, thr=thr))
    test_md5 = hashlib.md5((root / "data/splits" / a.split / "test.csv").read_bytes()).hexdigest()
    json.dump({"split": a.split, "test_md5": test_md5, "n_test": len(test), "n_pos": npos, "random_AUPRC": rand_auprc, "reference": a.ref,
               "bootstrap": {"n": a.boot, "seed": 42, "strata": "disease family x label" if strata is not None else "label"},
               "bonferroni": {"comparisons": k, "threshold": thr}, "flags": flags, "results": res}, open(out / "metrics.json", "w"), indent=1)
    print(table.to_string(index=False)); print(f"random-chance AUPRC = {npos}/{len(test)} = {rand_auprc:.4f}"); [print(f) for f in flags]


if __name__ == "__main__":
    main()
