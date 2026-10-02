"""Step 4, items 3-4: hallucination and path-faithfulness on two defined sets (Table R-D), with whole-number counts.

DEFINITIONS (fixed once, written into the manuscript)
  Entity        a specific biomedical entity mentioned in an explanation, found by (a) exact matching against the KG entity
                vocabulary and (b) BioBERT token-classification NER (chemical, disease, gene models). Generic words
                ("protein", "pathway", "receptor", ...) are never entities.
  Supported     the entity (normalised: lower case, punctuation stripped, trailing "pathway/receptor/protein/inhibitor(s)"
                removed) equals a node name on a retrieved evidence path of that pair, an alias of one (--aliases), or the query
                drug / disease of that pair.
  Hallucinated explanation   contains at least one specific entity that is not supported.
  Hallucination rate         k / N, k = hallucinated explanations, N = explanations of that model on that set.
  Path faithfulness          supported entities / all entities, pooled over the explanations of a set (entity level).
  Specificity                share of unsupported entities that are database entities (found in the KG vocabulary).
  Sensitivity analysis       the same rate using KG-vocabulary matches only (no NER), so the result does not depend on NER errors.
Reference evidence for every model is the SAME file (evidence.jsonl from build_evidence.py). Explanations that are missing
are counted in N_missing and NOT silently dropped.

Subcommands
  run       python -m exdrkg_rerun.explainability.hallucination_eval run --vocab data/kg_entity_names.json --aliases data/aliases.json
  validate  python -m exdrkg_rerun.explainability.hallucination_eval validate --sheet results/rerun/hallucination/manual_validation_100.csv
            (after annotators fill columns entity_is_real and unsupported_is_correct; reports NER/flag precision as required by the plan)
Outputs (results/rerun/hallucination/): R-D.csv, R-D.md, R-D_detail.json, per_explanation.csv, manual_validation_100.csv, pairwise_mcnemar.csv
"""
import argparse, itertools, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

GENERIC = {"protein", "proteins", "gene", "genes", "pathway", "pathways", "receptor", "receptors", "enzyme", "enzymes", "disease", "diseases", "drug", "drugs",
           "cell", "cells", "signaling", "inhibitor", "inhibitors", "kinase", "kinases", "channel", "mechanism", "patient", "patients", "treatment", "therapy",
           "cancer", "inflammation", "tumor", "syndrome", "disorder", "target", "targets", "pathophysiology", "expression", "activity", "binding"}
NER_DEFAULT = ["alvaroalon2/biobert_chemical_ner", "alvaroalon2/biobert_diseases_ner", "alvaroalon2/biobert_genetic_ner"]
SUFFIX = re.compile(r"\s+(pathway|pathways|receptor|receptors|protein|proteins|inhibitor|inhibitors|signaling|signalling)$")


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s\-/]", " ", s.lower()); s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s: prev = s; s = SUFFIX.sub("", s)
    return s


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def build_vocab_matcher(vocab):
    names = sorted({v for v in vocab if norm(v) not in GENERIC and len(norm(v)) > 2}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.I) if names else None


def make_ner(models, device):
    from transformers import pipeline
    pipes = [pipeline("token-classification", model=m, aggregation_strategy="simple", device=device) for m in models]
    def run(text):
        spans = []
        for p in pipes:
            for s in p(text):
                w = s["word"].replace(" ##", "").replace("##", "").strip()
                if s["score"] >= 0.80 and len(w) > 2 and norm(w) not in GENERIC: spans.append(w)
        return spans
    return run


def extract(text, rx, ner):
    kg = {norm(m.group(1)) for m in rx.finditer(text)} if rx else set()
    kg = {e for e in kg if e and e not in GENERIC}
    nn_ = {norm(w) for w in ner(text)} if ner else set()
    nn_ = {e for e in nn_ if e and e not in GENERIC}
    merged = set(kg)
    for e in nn_:                       # drop NER spans that are fragments of a longer KG match
        if not any(e != k and (e in k) for k in kg): merged.add(e)
    return merged, kg


def supported(e, ev_set):
    return e in ev_set


def cmd_run(a):
    root = Path(a.root); sd = root / a.sets; info = json.load(open(sd / "eval_sets.json"))
    ev = {j["pair_id"]: j for j in map(json.loads, open(sd / "evidence.jsonl"))}
    aliases = json.load(open(a.aliases)) if a.aliases else {}
    def evidence_set(pid):
        base = {norm(e) for e in ev[pid]["entities"]}
        for e in list(base):
            for al in aliases.get(e, []): base.add(norm(al))
        return base
    rx = build_vocab_matcher(json.load(open(a.vocab))); ner = None if a.no_ner else make_ner(a.ner_models, 0 if a.device.startswith("cuda") else -1)
    rows = []
    for j in map(json.loads, open(sd / "explanations.jsonl")):
        if j["pair_id"] not in ev: continue
        ents, kg_only = extract(j["explanation"], rx, ner); es = evidence_set(j["pair_id"])
        bad = sorted(e for e in ents if not supported(e, es)); bad_kg = sorted(e for e in kg_only if not supported(e, es))
        rows.append({"model": j["model"], "set": j["set"], "pair_id": j["pair_id"], "n_entities": len(ents), "n_unsupported": len(bad),
                     "hallucinated": int(len(bad) > 0), "hallucinated_kg_only": int(len(bad_kg) > 0), "n_kg_entities": len(kg_only),
                     "n_unsupported_kg": len(bad_kg), "unsupported": ";".join(bad), "entities": ";".join(sorted(ents)), "explanation": j["explanation"]})
    df = pd.DataFrame(rows); df.to_csv(sd / "per_explanation.csv", index=False)
    agg, table = {}, []
    order = a.models or sorted(df.model.unique())
    for m in order:
        rec = {"Model": m}
        for s in ("H1", "H2"):
            g = df[(df.model == m) & (df.set == s)]; N_exp = info[s]["N"]; n = len(g); k = int(g.hallucinated.sum())
            lo, hi = wilson(k, n); ent = int(g.n_entities.sum()); bad = int(g.n_unsupported.sum())
            k_kg = int(g.hallucinated_kg_only.sum())
            agg[(m, s)] = {"N_expected": N_exp, "N": n, "N_missing": N_exp - n, "k": k, "pct": round(100 * k / n, 1) if n else None, "wilson95": [round(lo, 1), round(hi, 1)],
                           "explanations_with_zero_entities": int((g.n_entities == 0).sum()), "mean_unsupported_per_explanation": round(g.n_unsupported.mean(), 3) if n else None,
                           "entities": ent, "unsupported_entities": bad, "path_faithfulness": round((ent - bad) / ent, 3) if ent else None,
                           "specificity_pct_of_unsupported_that_are_KG_entities": round(100 * g.n_unsupported_kg.sum() / bad, 1) if bad else None,
                           "kg_only_k": k_kg, "kg_only_pct": round(100 * k_kg / n, 1) if n else None}
            r = agg[(m, s)]
            rec[f"{s}: hallucinated k / N"] = f"{k} / {n}"; rec[f"{s}: %"] = r["pct"]
            if s == "H2": rec["Path faithfulness (H2)"] = r["path_faithfulness"]
        table.append(rec)
    T = pd.DataFrame(table); T.to_csv(sd / "R-D.csv", index=False)
    text = T.to_markdown(index=False) + f"\n\nH1: {info['H1']['definition']}\nH2: {info['H2']['definition']}\n"
    (sd / "R-D.md").write_text(text)
    json.dump({f"{m}|{s}": v for (m, s), v in agg.items()}, open(sd / "R-D_detail.json", "w"), indent=1)
    # consistency: k / N must reproduce the percentage
    for (m, s), v in agg.items():
        if v["N"] and abs(100 * v["k"] / v["N"] - v["pct"]) > 0.051: raise SystemExit(f"k/N does not reproduce % for {m} {s}")
    # paired exact McNemar between models on the same pairs (uses hallucinated flag)
    mc = []
    for s in ("H1", "H2"):
        piv = df[df.set == s].pivot_table(index="pair_id", columns="model", values="hallucinated", aggfunc="max")
        pairs = list(itertools.combinations([m for m in order if m in piv.columns], 2)); thr = 0.05 / max(len(pairs), 1)
        for x, y in pairs:
            both = piv[[x, y]].dropna(); b = int(((both[x] == 1) & (both[y] == 0)).sum()); c = int(((both[x] == 0) & (both[y] == 1)).sum())
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            mc.append({"set": s, "model_a": x, "model_b": y, "n_pairs": len(both), "a_only_hallucinates": b, "b_only_hallucinates": c, "p_mcnemar_exact": p, "bonferroni_threshold": thr, "significant": bool(p < thr)})
    pd.DataFrame(mc).to_csv(sd / "pairwise_mcnemar.csv", index=False)
    # 100 random explanations for manual validation (stratified by model, seed 42)
    rng = np.random.default_rng(42); per = max(1, 100 // max(df.model.nunique(), 1)); parts = []
    for m, g in df[df.n_entities > 0].groupby("model"): parts.append(g.sample(min(per, len(g)), random_state=42))
    V = pd.concat(parts).head(100)[["model", "set", "pair_id", "explanation", "entities", "unsupported"]].assign(entity_is_real="", unsupported_is_correct="")
    V.to_csv(sd / "manual_validation_100.csv", index=False)
    print(text); print(pd.DataFrame(mc).to_string(index=False))


def cmd_validate(a):
    v = pd.read_csv(a.sheet)
    ok = v[v.entity_is_real.astype(str).str.strip().isin(["0", "1"])]
    if ok.empty: raise SystemExit("fill entity_is_real and unsupported_is_correct (1/0) first")
    ner_prec = ok.entity_is_real.astype(int).mean(); flag = ok[ok.unsupported.fillna("").astype(str) != ""]
    flag_prec = flag.unsupported_is_correct.astype(int).mean() if len(flag) else float("nan")
    print(f"validated explanations: {len(ok)}\nentity extraction precision: {ner_prec:.3f}\nprecision of 'unsupported' flags: {flag_prec:.3f} (n = {len(flag)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--root", default="."); r.add_argument("--sets", default="results/rerun/hallucination")
    r.add_argument("--vocab", required=True); r.add_argument("--aliases", default=None); r.add_argument("--ner-models", nargs="*", default=NER_DEFAULT)
    r.add_argument("--no-ner", action="store_true"); r.add_argument("--device", default="cuda"); r.add_argument("--models", nargs="*", default=None)
    v = sub.add_parser("validate"); v.add_argument("--sheet", required=True)
    a = ap.parse_args(); cmd_run(a) if a.cmd == "run" else cmd_validate(a)
