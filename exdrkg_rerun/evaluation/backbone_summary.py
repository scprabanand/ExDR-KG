"""Step 10 (Comment 2.6): Table R-G, LLM backbone replacement (Stage 2, V4 configuration, no RLHF, identical settings).

Rows = backbones (BioMistral-7B V4 reference, one recent general-purpose model, one biomedical model). Columns:
  exact model name + commit hash (from run.json), parameters (total / trainable LoRA), Stage-2 hours, peak GPU GB, test AUROC/AUPRC/Hit@10/MRR with
  95% CIs, DeLong p vs BioMistral V4 on AUROC, H1 hallucination k / N (from Step 4's R-D_detail.json), and a SETTINGS CHECK that every run used the same
  data md5, LoRA config, epochs, lr, lambda_faith, seed (a mismatch stops the script: the comparison is only valid with identical settings).
Usage python -m exdrkg_rerun.evaluation.backbone_summary --root . --runs "BioMistral-7B=models/stage2_BioMistral_BioMistral-7B_V4" "Llama-3.1-8B=models/stage2_meta-llama_Llama-3.1-8B-Instruct_V4" ...
      (each --runs entry is <score-file-name>=<adapter folder>; the score CSV must be results/rerun/main/scores/<name>.csv)
Outputs results/rerun/backbones/R-G.{csv,md,tex}, settings_check.json
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--reference", default=None)
    ap.add_argument("--scores", default="results/rerun/main/scores"); ap.add_argument("--hallu", default="results/rerun/hallucination/R-D_detail.json"); ap.add_argument("--out", default="results/rerun/backbones")
    ap.add_argument("--external-ref-scores", default=None, help="plan rule: BioMistral-7B in R-G = V4 in R-B. Give the V4 score csv (from Step 3) to use the EXISTING V4 as the reference row instead of a retrained run")
    ap.add_argument("--external-ref-hallu", default="results/rerun/hallucination/R-D_detail.json"); ap.add_argument("--external-ref-hallu-name", default="V4"); ap.add_argument("--external-ref-hours", type=float, default=None); ap.add_argument("--external-ref-gpu-gb", type=float, default=None)
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    runs = [r.split("=", 1) for r in a.runs]; ext = "BioMistral-7B (= V4 in R-B)" if a.external_ref_scores else None; ref = ext or a.reference or runs[0][0]; S, R = {}, {}
    for n, d in runs: S[n] = pd.read_csv(root / a.scores / f"{n}.csv"); R[n] = json.load(open(root / d / "run.json"))
    if ext: S[ext] = pd.read_csv(root / a.external_ref_scores)
    keys = ["data_md5", "lora", "epochs", "lr", "lambda_faith", "lambda_lm", "seed", "faithfulness", "rlhf"]
    chk = {k: {n: R[n].get(k) for n in R} for k in keys}; bad = [k for k, v in chk.items() if len({json.dumps(x, sort_keys=True) for x in v.values()}) > 1]
    json.dump({"checked": chk, "mismatched": bad, "reference": "external existing V4 (settings not re-checked; same as R-B row V4)" if a.external_ref_scores else "first run"}, open(out / "settings_check.json", "w"), indent=1)
    if bad: raise SystemExit(f"settings differ across backbones: {bad} (see settings_check.json); the comparison would not be like-for-like")
    hd = json.load(open(root / a.hallu)) if (root / a.hallu).exists() else {}
    for n in S:
        if not S[n][["drug_id", "disease_id"]].astype(str).equals(S[ref][["drug_id", "disease_id"]].astype(str)): raise SystemExit(f"{n}: pairs/order differ from {ref}")
    rows = []
    if ext:
        s_ = S[ext]; m = compute_metrics(s_, s_.score.values); ci = bootstrap_ci(s_, s_.score.values); hx = (json.load(open(root / a.external_ref_hallu)) if (root / a.external_ref_hallu).exists() else {}).get(f"{a.external_ref_hallu_name}|H1", {})
        rows.append({"Backbone": ext, "commit": "existing checkpoint", "run name": a.external_ref_scores, "Params (B)": None, "LoRA params (M)": None, "Stage 2 (h)": a.external_ref_hours, "Peak GPU (GB)": a.external_ref_gpu_gb,
                     "AUROC": m["AUROC"], "AUROC 95% CI": str(ci.get("AUROC")), "AUPRC": m["AUPRC"], "Hit@10": m.get("Hit@10"), "MRR": m.get("MRR"), "DeLong p vs ref (AUROC)": None,
                     "H1 hallucinated k/N": f"{hx.get('k')} / {hx.get('N')}" if hx else "not run (Step 4 H1)", "H1 %": hx.get("pct")})
    for n, d in runs:
        s = S[n]; m = compute_metrics(s, s.score.values); ci = bootstrap_ci(s, s.score.values); r = R[n]
        p = None if n == ref else delong_test(s.label.values, s.score.values, S[ref].score.values)[2]
        h = hd.get(f"{n}|H1") or hd.get(f"{r['model']}|H1") or {}
        rows.append({"Backbone": r["model"], "commit": r.get("commit_hash"), "run name": n, "Params (B)": round(r["total_params"] / 1e9, 2), "LoRA params (M)": round(r["trainable_params"] / 1e6, 2),
                     "Stage 2 (h)": round(r["stage2_hours"], 2), "Peak GPU (GB)": round(r["peak_gpu_gb"], 1), "AUROC": m["AUROC"], "AUROC 95% CI": str(ci.get("AUROC")), "AUPRC": m["AUPRC"],
                     "Hit@10": m.get("Hit@10"), "MRR": m.get("MRR"), "DeLong p vs ref (AUROC)": p,
                     "H1 hallucinated k/N": f"{h.get('k')} / {h.get('N')}" if h else "not run (Step 4 H1)", "H1 %": h.get("pct")})
    T = pd.DataFrame(rows); T.to_csv(out / "R-G.csv", index=False); (out / "R-G.md").write_text(T.to_markdown(index=False)); (out / "R-G.tex").write_text(T.to_latex(index=False)); print(T.to_string(index=False))


if __name__ == "__main__":
    main()
