"""Step 3: ablation V1-V5 on the held-out test set (Table R-B), from per-variant score files.

Reads results/rerun/ablation/scores/{V1,V2,V3,V4,V5}.csv (columns drug_id, disease_id, label, score; produced by
score_ablation_variants.py) and computes, from those files only, exactly the metrics of Step 2 (metrics.py):
AUROC, AUPRC (pooled), Hit@1, Hit@10, MRR, 95% bootstrap CIs (1,000 resamples, seed 42, stratified by disease family x
label when --family-map is given).

Each SUCCESSIVE step (V1 vs V2, V2 vs V3, V3 vs V4, V4 vs V5) is tested with the DeLong test for AUROC and a paired
bootstrap (identical resamples for both variants) for AUPRC and Hit@10, Bonferroni-corrected for 4 comparisons
(threshold 0.05 / 4 = 0.0125). The script then writes plain-language findings that follow the data: a step is called
significant only if its p-value passes the corrected threshold, and a step that is not significant, or that goes down, is
reported as such (the manuscript must not claim that all increments are significant unless they are here).

Consistency checks required by the rerun plan
  * V5 must equal the ExDR-KG row of Table R-A (same scores): compared against results/rerun/main/scores/ExDR-KG.csv;
  * every variant must score the same test pairs (aligned on the pair key, labels checked).

Usage
  python -m exdrkg_rerun.evaluation.ablation_evaluation --root . --split standard_degree_matched --family-map data/disease_family.csv
Outputs (results/rerun/ablation/): table_RB.csv, table_RB.md, table_RB.tex, steps.csv, findings.txt, metrics.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test, paired_bootstrap_p
from step2_main_evaluation import load_scores

VARIANTS = ["V1", "V2", "V3", "V4", "V5"]
LABELS = {"V1": "V1 Encoder + MLP", "V2": "V2 + Path Retriever", "V3": "V3 + LLM (no faithfulness)",
          "V4": "V4 + Faithfulness loss", "V5": "V5 + Explainer + RLHF"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard")
    ap.add_argument("--scores", default="results/rerun/ablation/scores"); ap.add_argument("--out", default="results/rerun/ablation")
    ap.add_argument("--main-scores", default="results/rerun/main/scores", help="Table R-A score files, for the V5 = ExDR-KG check")
    ap.add_argument("--family-map", default=None); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(root / "data/splits" / a.split / "test.csv").reset_index(drop=True)
    if test.duplicated(["drug_id", "disease_id"]).any(): raise SystemExit("test file has repeated pairs; run rebuild_splits.py first")
    strata = None
    if a.family_map:
        fam = pd.read_csv(root / a.family_map).set_index("disease_id").family; strata = test.disease_id.map(fam).fillna("Other").values
    else:
        print("WARNING: no --family-map; bootstrap stratified by label only (state this in the manuscript)")
    missing = [v for v in VARIANTS if not (root / a.scores / f"{v}.csv").exists()]
    if missing: raise SystemExit(f"missing score files for {missing} in {a.scores}")
    S = {v: load_scores(root / a.scores / f"{v}.csv", test, v) for v in VARIANTS}
    res = {v: {"metrics": compute_metrics(test, S[v]), "ci95": bootstrap_ci(test, S[v], n=a.boot, strata=strata)} for v in VARIANTS}
    n_cmp = len(VARIANTS) - 1; thr = 0.05 / n_cmp
    steps = []
    for prev, cur in zip(VARIANTS[:-1], VARIANTS[1:]):
        auc_p, auc_c, p_auc = delong_test(test.label.values, S[cur], S[prev])
        p_prc = paired_bootstrap_p(test, S[cur], S[prev], "AUPRC", n=a.boot, strata=strata)
        p_h10 = paired_bootstrap_p(test, S[cur], S[prev], "Hit@10", n=a.boot, strata=strata)
        mp, mc = res[prev]["metrics"], res[cur]["metrics"]
        steps.append({"step": f"{prev}->{cur}", "d_AUROC": mc["AUROC"] - mp["AUROC"], "p_AUROC_delong": p_auc,
                      "d_AUPRC": mc["AUPRC"] - mp["AUPRC"], "p_AUPRC_boot": p_prc, "d_Hit@10": mc["Hit@10"] - mp["Hit@10"], "p_Hit@10_boot": p_h10,
                      "threshold": thr, "AUROC_sig": bool(p_auc < thr), "AUPRC_sig": bool(p_prc < thr), "Hit@10_sig": bool(p_h10 < thr)})
    steps_df = pd.DataFrame(steps); steps_df.to_csv(out / "steps.csv", index=False)
    rows = []
    for i, v in enumerate(VARIANTS):
        m, ci = res[v]["metrics"], res[v]["ci95"]
        rows.append({"Variant": LABELS[v], "AUROC (95% CI)": f"{m['AUROC']:.3f} ({ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f})",
                     "AUPRC": f"{m['AUPRC']:.3f}", "Hit@1": f"{m['Hit@1']:.3f}", "Hit@10": f"{m['Hit@10']:.3f}", "MRR": f"{m['MRR']:.3f}",
                     "p vs previous (AUROC)": "-" if i == 0 else f"{steps[i - 1]['p_AUROC_delong']:.4g}"})
    table = pd.DataFrame(rows); npos = int(test.label.sum())
    # ---- consistency with Table R-A
    checks = []
    ra = root / a.main_scores / "ExDR-KG.csv"
    if ra.exists():
        same = np.allclose(load_scores(ra, test, "ExDR-KG"), S["V5"], rtol=0, atol=1e-12)
        checks.append("V5 scores are identical to the ExDR-KG row of Table R-A." if same else
                      "CHECK FAILED: V5 scores differ from the ExDR-KG row of Table R-A. They must be the same model on the same test pairs; re-score both with score_ablation_variants.py.")
    else:
        checks.append(f"NOT CHECKED: {a.main_scores}/ExDR-KG.csv not found, so V5 = ExDR-KG could not be verified.")
    # ---- findings that follow the data
    lines = []
    for s in steps:
        d, name = s["d_AUROC"], s["step"]
        direction = "increased" if d > 0 else ("decreased" if d < 0 else "did not change")
        sig = "significant" if s["AUROC_sig"] else "NOT significant"
        lines.append(f"{name}: AUROC {direction} by {abs(d):.3f} ({sig} after Bonferroni, DeLong p = {s['p_AUROC_delong']:.3g}); "
                     f"AUPRC {s['d_AUPRC']:+.3f} (p = {s['p_AUPRC_boot']:.3g}, {'significant' if s['AUPRC_sig'] else 'not significant'}); "
                     f"Hit@10 {s['d_Hit@10']:+.3f} (p = {s['p_Hit@10_boot']:.3g}, {'significant' if s['Hit@10_sig'] else 'not significant'}).")
    all_sig = all(s["AUROC_sig"] and s["AUPRC_sig"] and s["Hit@10_sig"] for s in steps)
    mono = all(s["d_AUROC"] > 0 for s in steps)
    lines.append("Every step significant on all three metrics: " + ("YES" if all_sig else "NO - the manuscript must not state that all incremental improvements are significant."))
    lines.append("AUROC increases monotonically from V1 to V5: " + ("yes" if mono else "NO - report the steps that decrease."))
    lines.append(f"Random-chance AUPRC = {npos}/{len(test)} = {npos / len(test):.4f}.")
    (out / "findings.txt").write_text("\n".join(lines + [""] + checks))
    table.to_csv(out / "table_RB.csv", index=False); (out / "table_RB.md").write_text(table.to_markdown(index=False) + "\n\n" + "\n".join(lines + [""] + checks)); (out / "table_RB.tex").write_text(table.to_latex(index=False, escape=True))
    json.dump({"split": a.split, "test_md5": hashlib.md5((root / "data/splits" / a.split / "test.csv").read_bytes()).hexdigest(),
               "bonferroni": {"comparisons": n_cmp, "threshold": thr}, "bootstrap": {"n": a.boot, "seed": 42, "strata": "family x label" if strata is not None else "label"},
               "results": res, "steps": steps, "checks": checks}, open(out / "metrics.json", "w"), indent=1)
    print(table.to_string(index=False)); print(); print("\n".join(lines)); print(); print("\n".join(checks))


if __name__ == "__main__":
    main()
