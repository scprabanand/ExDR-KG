"""Step 11: Table R-H (updated LLM baselines vs ExDR-KG on the SAME pairs) and Table R-K (RAG configuration and run record).

R-H: for every model score file the script restricts ALL models to the common pair set (the full 11,615 test pairs, or the fixed sample when a sample
     was used), then reports AUROC, AUPRC (with the prevalence-matched random AUPRC), Hit@10, MRR with 95% bootstrap CIs, DeLong p vs the reference
     (ExDR-KG), the number of unanswered/unparsed pairs, and tied-score share. When a negative sample is used, ranking metrics (Hit@k, MRR) are computed
     within the sample and this is stated in the table note; AUPRC depends on prevalence, so ExDR-KG is re-evaluated on the same sample.
R-K: RAG configuration (from --rag-config json), model string(s) actually served, query dates, prompt md5, token counts, API errors.
Usage
  python -m exdrkg_rerun.evaluation.llm_baseline_summary --root . --reference ExDR-KG --models "GPT-4o+RAG" "GPT-4o+ExDR-KG paths (zero-shot)" GPT-4+RAG \\
      --scores results/rerun/main/scores [--sample results/rerun/llm/sample_ids.csv] --runs results/rerun/llm/run_gpt-4o-2024-08-06_rag.json results/rerun/llm/run_gpt-4o-2024-08-06_paths.json \\
      --rag-config data/rag_config.json
Outputs results/rerun/llm/R-H.{csv,md,tex}, R-K_rag_config_and_runs.md/.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test


def flat(d, p=""):
    for k, v in d.items():
        if k.startswith("_"): continue
        if isinstance(v, dict): yield from flat(v, p + k + ".")
        else: yield p + k, v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--reference", default="ExDR-KG"); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scores", default="results/rerun/main/scores"); ap.add_argument("--sample", default=None); ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--rag-config", default=None); ap.add_argument("--hallu", default="results/rerun/hallucination_llm/R-D_detail.json", help="Step 4 detail json holding <model>|H1 counts"); ap.add_argument("--out", default="results/rerun/llm"); a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    names = [a.reference] + a.models; sd = root / a.scores; S = {}
    for n in names:
        p = next((c for c in (sd / f"{n}.csv", root / "results/rerun/main/scores" / f"{n}.csv", root / "results/rerun/main/sample_scores" / f"{n}.csv") if c.exists()), None)
        if p is None: raise SystemExit(f"no score file for {n}")
        df = pd.read_csv(p); df["pair"] = df.drug_id.astype(str) + "|" + df.disease_id.astype(str); S[n] = df.drop_duplicates("pair").set_index("pair")
    common = set.intersection(*[set(v.index) for v in S.values()])
    if a.sample: common &= {l.strip() for l in open(root / a.sample) if l.strip() and l.strip() != "pair"}
    common = sorted(common); print(f"common pairs: {len(common)}"); base = S[a.reference].loc[common].reset_index()
    prev = float(base.label.mean()); rows = []
    for n in names:
        d = S[n].loc[common].reset_index(); assert (d.label.values == base.label.values).all(), f"labels differ for {n}"
        m = compute_metrics(d, d.score.values); ci = bootstrap_ci(d, d.score.values); p = None if n == a.reference else delong_test(d.label.values, d.score.values, base.score.values)[2]
        hd_ = json.load(open(root / a.hallu)) if (root / a.hallu).exists() else {}; h1 = hd_.get(f"{n}|H1", {})
        ctx = "ExDR-KG top-10 paths, zero-shot" if "paths" in n.lower() else ("(reference: scoring pass, Eq. 22)" if n == a.reference else "Text RAG (PubMed + DrugBank)")
        rows.append({"Model": n, "Context given": ctx, "N pairs": len(d), "positives": int(d.label.sum()), "AUROC": m["AUROC"], "AUROC 95% CI": str(ci.get("AUROC")), "AUPRC": m["AUPRC"], "AUPRC 95% CI": str(ci.get("AUPRC")),
                     "AUPRC random": prev, "Hit@10": m.get("Hit@10"), "MRR": m.get("MRR"), "tied at 0.5 (%)": round(100 * float((d.score == 0.5).mean()), 2),
                     "DeLong p vs ref": p, "H1 hallucinated k / N": (f"{h1.get('k')} / {h1.get('N')}" if h1 else "not run (needs Step 4 H1 explanations)")})
    T = pd.DataFrame(rows); T.to_csv(out / "R-H.csv", index=False)
    note = (f"\n\nEvaluated on {len(common)} pairs ({int(base.label.sum())} positives, prevalence {prev:.4f}). " + ("A fixed random negative sample was used (same for every model); AUPRC and ranking metrics are computed within it and are not comparable with full-test numbers." if a.sample else "Full test set."))
    (out / "R-H.md").write_text(T.round(4).to_markdown(index=False) + note); (out / "R-H.tex").write_text(T.round(4).to_latex(index=False)); print(T.round(4).to_string(index=False)); print(note)
    rk = {"runs": [json.load(open(root / r)) for r in a.runs], "rag_config": None, "missing_rag_fields": []}
    if a.rag_config:
        cfg = json.load(open(root / a.rag_config)); rk["rag_config"] = cfg; rk["missing_rag_fields"] = [k for k, v in flat(cfg) if v is None]
    json.dump(rk, open(out / "R-K_rag_config_and_runs.json", "w"), indent=1)
    md = ["# Table R-K: run record and RAG configuration", ""]
    for r in rk["runs"]: md.append(f"- {r['model_requested']} [{r['context']}] served as {r['served_model_ids']} fingerprint {r['system_fingerprints']}; queried {r['query_first_utc']} to {r['query_last_utc']}; temperature 0, seed {r['seed']}; "
                                     f"{r['n_pairs']} pairs ({r['n_positives']} positives); API errors {r['api_errors']}; unparsed {r['unparsed']}; empty context {r['pairs_with_empty_context']}; prompt md5 {r['prompt_md5']}; tokens {r['prompt_tokens']}/{r['completion_tokens']}")
    if rk["rag_config"]:
        md += ["", "| field | value |", "|---|---|"] + [f"| {k} | {'NOT RECORDED' if v is None else v} |" for k, v in flat(rk["rag_config"])]
        if rk["missing_rag_fields"]: md.append(f"\nFields not recorded: {', '.join(rk['missing_rag_fields'])}. The response letter must not claim these were fixed unless they are recovered from the original pipeline.")
    (out / "R-K_rag_config_and_runs.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
