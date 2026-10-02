"""Step 7 (Comment 1.5): temporally novel case studies, Table R-F.

Selection logic (fixed before looking at explanations; every rule is written to results/rerun/cases/selection_log.md):
  1. Candidates = temporal TEST POSITIVES with first-evidence year >= --min-year (default 2020), scored by the temporal-split ExDR-KG
     model (the model trained on the pre-2020 HKG). Ranked by predicted score, highest first.
  2. A candidate is ELIGIBLE only if
       a) the disease id is a MONDO id (or is mapped to one with --disease-map) and the disease is specific (not in --exclude-diseases,
          not a generic family word);
       b) PROGRAMMATIC ABSENCE CHECK on the training HKG (the pre-2020 graph the model was trained on): no edge of ANY relation between
          the drug and the disease, in the forward direction, the inverse direction, or as a duplicate. Result recorded per candidate;
       c) a curated literature row exists in --literature (case-specific reference with year >= min-year; the script never invents one).
  3. Success cases = the first --n-success (5) eligible candidates in score order.
     Failure cases  = --n-fail (2) eligible candidates ranked in the bottom --fail-quantile (default lower 25 %) of the temporal test
                      positives, lowest score first; a short reason is taken from --literature (column failure_reason) or a
                      data-derived reason (no path / long path / low-degree disease) is written and must be reviewed.
  4. For each case: top evidence path (from evidence.jsonl if given, else the best directed path <= 4 hops on the training HKG:
     fewest hops, then lowest sum of log-degree of intermediates), the Level-2 explanation (from explanations.jsonl, model given by
     --explainer-model), the literature reference, and the absence-check result.
Inputs
  --scores      per-pair scores of the temporal-split ExDR-KG model (csv with drug id, disease id, label, score columns; names auto-detected)
  --test        temporal test file; needs a first-evidence year column (default names: first_evidence_year, year, first_year) or --years csv
  --train-kg    the graph the temporal model was trained on (pre-2020, masked: no val/test indication edges)
  --literature  csv you fill from PubMed / ClinicalTrials.gov / approval documents. Columns:
                drug_id, disease_id, drug_name, disease_name, mondo_id, reference_type (approval|trial|pubmed), reference_id (e.g. NCT..., PMID...),
                reference_year, mechanism_note, mechanism_agrees (yes/partly/no), failure_reason (failure cases only, optional)
  --evidence    optional evidence.jsonl from build_evidence.py (pair_id + paths)
  --explanations optional explanations.jsonl (pair_id, model, explanation)
Usage
  python -m exdrkg_rerun.explainability.case_studies --root . --scores results/rerun/main/scores/ExDR-KG_temporal.csv \\
      --test data/splits/temporal/test.csv --train-kg data/processed/masked_temporal.pt \\
      --literature data/case_literature.csv --explanations results/rerun/cases/explanations.jsonl
Run once with --list-candidates first: it writes the top candidates that pass the absence check so you can look up their references.
"""
import argparse, json, math, re
from collections import defaultdict, deque
from pathlib import Path
import numpy as np, pandas as pd, torch
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

GENERIC = {"cancer", "carcinoma", "tumor", "tumour", "neoplasm", "disease", "disorder", "syndrome", "infection", "inflammation", "diabetes mellitus",
           "cardiovascular disease", "neurodegenerative disease", "autoimmune disease", "metabolic disease", "genetic disease", "rare disease"}
COLS = {"drug": ["drug_id", "drug", "head", "src"], "dis": ["disease_id", "disease", "tail", "dst"], "label": ["label", "y"],
        "score": ["score", "prob", "pred", "s"], "year": ["first_evidence_year", "year", "first_year"]}


def col(df, key, given=None):
    if given: return given
    for c in COLS[key]:
        if c in df.columns: return c
    raise SystemExit(f"cannot find a {key} column in {list(df.columns)}; pass it explicitly")


def find_type(kg, word):
    m = [t for t in kg.node_types if word in t.lower()]
    if not m: raise SystemExit(f"no node type containing '{word}': {list(kg.node_types)}")
    return m[0]


def load_kg(kg_path):
    kg = torch.load(kg_path, weights_only=False, map_location="cpu")
    return kg


def absence_check(kg, dtype, stype, drug_ids, dis_ids):
    """For each (drug, disease) return dict with edges found in any relation, forward, inverse, and duplicates."""
    idx = {t: {str(r): i for i, r in enumerate(kg[t].raw_ids)} for t in (dtype, stype)}
    rels = [et for et in kg.edge_types if {et[0], et[2]} == {dtype, stype} or (et[0] == dtype and et[2] == stype) or (et[0] == stype and et[2] == dtype)]
    tabs = {}
    for et in rels:
        ei = kg[et].edge_index.cpu().numpy()
        tabs[et] = ei
    out = []
    for d, s in zip(drug_ids, dis_ids):
        di, si = idx[dtype].get(str(d)), idx[stype].get(str(s))
        rec = {"drug_in_graph": di is not None, "disease_in_graph": si is not None, "forward_edges": 0, "inverse_edges": 0, "duplicate_edges": 0, "relations_found": []}
        if di is not None and si is not None:
            for et, ei in tabs.items():
                if et[0] == dtype and et[2] == stype: n = int(((ei[0] == di) & (ei[1] == si)).sum()); rec["forward_edges"] += n
                elif et[0] == stype and et[2] == dtype: n = int(((ei[0] == si) & (ei[1] == di)).sum()); rec["inverse_edges"] += n
                else: n = 0
                if n: rec["relations_found"].append(f"{'|'.join(et)} x{n}"); rec["duplicate_edges"] += max(0, n - 1)
        rec["absent"] = bool(rec["drug_in_graph"] and rec["disease_in_graph"] and rec["forward_edges"] == 0 and rec["inverse_edges"] == 0)
        rec["relations_found"] = "; ".join(rec["relations_found"])
        out.append(rec)
    return out


class PathFinder:
    """Fallback top-path: best directed path <= max_hops (fewest hops, then lowest sum log-degree of intermediates)."""
    def __init__(self, kg, max_hops=4):
        self.kg, self.h = kg, max_hops; off, o = {}, 0
        for t in kg.node_types: off[t] = o; o += kg[t].num_nodes
        self.off, self.n = off, o; self.names = {}; self.type_of = np.empty(o, dtype=object)
        for t in kg.node_types:
            self.type_of[off[t]:off[t] + kg[t].num_nodes] = t
            for i, r in enumerate(kg[t].raw_ids): self.names[off[t] + i] = str(r)
        adj = defaultdict(list); deg = np.zeros(o, dtype=np.int64)
        for et in kg.edge_types:
            ei = kg[et].edge_index.cpu().numpy(); s, d = ei[0] + off[et[0]], ei[1] + off[et[2]]
            for a, b in zip(s.tolist(), d.tolist()): adj[a].append((b, et[1]))
            deg += np.bincount(s, minlength=o) + np.bincount(d, minlength=o)
        self.adj, self.deg = adj, deg

    def best(self, dtype, stype, drug_raw, dis_raw):
        idx = lambda t, r: {str(x): i for i, x in enumerate(self.kg[t].raw_ids)}.get(str(r))
        di, si = idx(dtype, drug_raw), idx(stype, dis_raw)
        if di is None or si is None: return None
        s, t = di + self.off[dtype], si + self.off[stype]; best = None
        stack = [(s, [s], [])]
        while stack:
            u, nodes, rels = stack.pop()
            if len(rels) >= self.h: continue
            for v, r in self.adj.get(u, []):
                if v in nodes: continue
                if v == t:
                    cost = (len(rels) + 1, float(sum(math.log1p(self.deg[x]) for x in nodes[1:])))
                    if best is None or cost < best[0]: best = (cost, nodes + [v], rels + [r])
                elif len(rels) + 1 < self.h:
                    stack.append((v, nodes + [v], rels + [r]))
        if best is None: return None
        _, nodes, rels = best; parts = [f"{self.names[nodes[0]]}"]
        for r, n in zip(rels, nodes[1:]): parts.append(f"-[{r}]-> {self.names[n]} ({self.type_of[n]})")
        return {"text": " ".join(parts), "hops": len(rels), "source": "fallback_best_path"}


def evidence_top(evfile):
    out = {}
    if not evfile or not Path(evfile).exists(): return out
    for line in open(evfile):
        j = json.loads(line); ps = j.get("paths") or j.get("evidence_paths") or []
        if ps:
            p = ps[0]; txt = p if isinstance(p, str) else (p.get("text") or p.get("path") or json.dumps(p))
            out[j["pair_id"]] = {"text": txt, "hops": None, "source": "evidence.jsonl (retriever, rank 1)"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--scores", required=True); ap.add_argument("--test", required=True); ap.add_argument("--train-kg", required=True)
    ap.add_argument("--years", default=None, help="optional csv drug_id,disease_id,first_evidence_year"); ap.add_argument("--literature", default=None)
    ap.add_argument("--disease-map", default=None, help="optional csv disease_id,mondo_id,disease_name"); ap.add_argument("--exclude-diseases", default=None, help="text file, one id or name per line")
    ap.add_argument("--evidence", default=None); ap.add_argument("--explanations", default=None); ap.add_argument("--explainer-model", default="ExDR-KG")
    ap.add_argument("--min-year", type=int, default=2020); ap.add_argument("--n-success", type=int, default=5); ap.add_argument("--n-fail", type=int, default=2)
    ap.add_argument("--fail-quantile", type=float, default=0.25); ap.add_argument("--list-candidates", action="store_true"); ap.add_argument("--n-list", type=int, default=60)
    ap.add_argument("--out", default="results/rerun/cases"); ap.add_argument("--max-hops", type=int, default=4)
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)

    sc, te = pd.read_csv(root / a.scores), pd.read_csv(root / a.test)
    cd, cs, cl, cc = col(sc, "drug"), col(sc, "dis"), col(sc, "label"), col(sc, "score")
    sc = sc.rename(columns={cd: "drug_id", cs: "disease_id", cl: "label", cc: "score"})[["drug_id", "disease_id", "label", "score"]]
    sc[["drug_id", "disease_id"]] = sc[["drug_id", "disease_id"]].astype(str)
    if a.years:
        y = pd.read_csv(root / a.years).astype({"drug_id": str, "disease_id": str}); sc = sc.merge(y[["drug_id", "disease_id", "first_evidence_year"]], on=["drug_id", "disease_id"], how="left")
    else:
        td, ts, ty = col(te, "drug"), col(te, "dis"), col(te, "year")
        y = te.rename(columns={td: "drug_id", ts: "disease_id", ty: "first_evidence_year"})[["drug_id", "disease_id", "first_evidence_year"]].astype({"drug_id": str, "disease_id": str}).drop_duplicates(["drug_id", "disease_id"])
        sc = sc.merge(y, on=["drug_id", "disease_id"], how="left")
    if sc.first_evidence_year.isna().all(): raise SystemExit("no first-evidence years found: the temporal test file must carry them (rebuild it with dates); nothing is invented")
    sc["pair_id"] = sc.drug_id + "|" + sc.disease_id
    # rank of each pair inside its disease (all test rows, positives and negatives) and among all positives
    sc["rank_in_disease"] = sc.groupby("disease_id").score.rank(ascending=False, method="average")
    sc["n_candidates_in_disease"] = sc.groupby("disease_id").score.transform("size")
    pos = sc[(sc.label == 1) & (sc.first_evidence_year >= a.min_year)].sort_values("score", ascending=False).reset_index(drop=True)
    pos["rank_among_new_positives"] = np.arange(1, len(pos) + 1); pos["pct_rank"] = pos.rank_among_new_positives / len(pos)
    print(f"temporal test positives with first evidence >= {a.min_year}: {len(pos)}")

    kg = load_kg(root / a.train_kg); dtype, stype = find_type(kg, "drug"), find_type(kg, "disease")
    dm = pd.read_csv(root / a.disease_map).astype(str) if a.disease_map else None
    def mondo(r):
        if dm is not None and r in set(dm.disease_id): return dm.set_index("disease_id").loc[r, "mondo_id"]
        return r if re.match(r"^MONDO:\d+$", r) else None
    pos["mondo_id"] = pos.disease_id.map(mondo)
    ex = set(x.strip().lower() for x in open(root / a.exclude_diseases)) if a.exclude_diseases else set()
    pos["specific"] = pos.disease_id.str.lower().map(lambda x: x not in ex and x not in GENERIC) & pos.mondo_id.notna()
    chk = pd.DataFrame(absence_check(kg, dtype, stype, pos.drug_id, pos.disease_id)); pos = pd.concat([pos, chk], axis=1)
    lit = pd.read_csv(root / a.literature).astype(str) if a.literature and Path(root / a.literature).exists() else pd.DataFrame(columns=["drug_id", "disease_id"])
    if len(lit):
        lit["reference_year_n"] = pd.to_numeric(lit.get("reference_year"), errors="coerce"); lit = lit.drop_duplicates(["drug_id", "disease_id"])
        pos = pos.merge(lit, on=["drug_id", "disease_id"], how="left", suffixes=("", "_lit"))
        pos["has_literature"] = pos.reference_id.notna() & (pos.reference_year_n >= a.min_year)
    else: pos["has_literature"] = False
    pos["eligible"] = pos.specific & pos.absent & pos.has_literature
    pos.drop(columns=[c for c in ["explanation"] if c in pos]).to_csv(out / "candidates_absence_check.csv", index=False)

    if a.list_candidates or not len(lit):
        L = pos[pos.specific & pos.absent].head(a.n_list)[["rank_among_new_positives", "drug_id", "disease_id", "mondo_id", "score", "first_evidence_year", "rank_in_disease", "n_candidates_in_disease"]]
        L.to_csv(out / "candidate_list_for_literature.csv", index=False)
        print("candidates that pass the specificity and absence checks (look up NCT / PMID / approval for these and fill the literature csv):"); print(L.to_string(index=False))
        if not len(lit): return

    succ = pos[pos.eligible].head(a.n_success).assign(case_type="success")
    low = pos[pos.eligible & (pos.pct_rank >= 1 - a.fail_quantile)].sort_values("score").head(a.n_fail).assign(case_type="failure")
    cases = pd.concat([succ, low]).reset_index(drop=True)
    if len(succ) < a.n_success: print(f"WARNING: only {len(succ)} eligible success cases; add literature rows for more candidates")
    if len(low) < a.n_fail: print(f"WARNING: only {len(low)} eligible failure cases in the bottom {a.fail_quantile:.0%}")

    ev = evidence_top(root / a.evidence if a.evidence else None); pf = None; expl = {}
    if a.explanations and Path(root / a.explanations).exists():
        for line in open(root / a.explanations):
            j = json.loads(line)
            if j.get("model") == a.explainer_model: expl[j["pair_id"]] = j["explanation"]
    recs = []
    for _, r in cases.iterrows():
        p = ev.get(r.pair_id)
        if p is None:
            pf = pf or PathFinder(kg, a.max_hops); p = pf.best(dtype, stype, r.drug_id, r.disease_id) or {"text": f"NO directed path <= {a.max_hops} hops on the training HKG", "hops": None, "source": "fallback_best_path"}
        reason = r.get("failure_reason") if r.case_type == "failure" else None
        if r.case_type == "failure" and (reason is None or str(reason) in ("nan", "")):
            reason = "DATA-DERIVED (review): " + ("no evidence path within 4 hops" if p["text"].startswith("NO") else f"only a {p['hops']}-hop path" if p["hops"] and p["hops"] >= 3 else "score low despite path; check disease degree in R-E2")
        recs.append({"case_type": r.case_type, "drug_id": r.drug_id, "drug_name": r.get("drug_name"), "disease_id": r.disease_id, "mondo_id": r.mondo_id, "disease_name": r.get("disease_name"),
                     "first_evidence_year": int(r.first_evidence_year), "score": round(float(r.score), 4), "rank_among_new_positives": int(r.rank_among_new_positives), "n_new_positives": len(pos),
                     "rank_in_disease": float(r.rank_in_disease), "n_candidates_in_disease": int(r.n_candidates_in_disease),
                     "absent_from_training_HKG": bool(r.absent), "forward_edges": int(r.forward_edges), "inverse_edges": int(r.inverse_edges), "duplicate_edges": int(r.duplicate_edges),
                     "top_evidence_path": p["text"], "path_source": p["source"], "level2_explanation": expl.get(r.pair_id, "NOT GENERATED: run generate_explanations.py for this pair"),
                     "reference_type": r.get("reference_type"), "reference_id": r.get("reference_id"), "reference_year": r.get("reference_year"), "mechanism_note": r.get("mechanism_note"), "mechanism_agrees": r.get("mechanism_agrees", ""),
                     "failure_reason": reason})
    R = pd.DataFrame(recs); R.to_csv(out / "R-F.csv", index=False); json.dump(recs, open(out / "R-F_cases.json", "w"), indent=1)
    (out / "R-F.md").write_text(R[["case_type", "drug_id", "mondo_id", "first_evidence_year", "score", "rank_among_new_positives", "absent_from_training_HKG", "reference_type", "reference_id", "reference_year", "mechanism_agrees"]].to_markdown(index=False))
    log = [f"# Step 7 selection log", f"training HKG: {a.train_kg}; scores: {a.scores}; explainer: {a.explainer_model}", f"new positives (first evidence >= {a.min_year}): {len(pos)}",
           f"specific & MONDO: {int(pos.specific.sum())}; also absent from training HKG: {int((pos.specific & pos.absent).sum())}; also literature-backed: {int(pos.eligible.sum())}",
           f"success = first {a.n_success} eligible by score; failure = lowest-scoring eligible in bottom {a.fail_quantile:.0%}", "",
           "Rejected above the last chosen success case (reason):"]
    last = int(succ.rank_among_new_positives.max()) if len(succ) else 0
    for _, r in pos[pos.rank_among_new_positives <= last].iterrows():
        if not r.eligible: log.append(f"  #{int(r.rank_among_new_positives)} {r.drug_id} - {r.disease_id}: " + ", ".join(k for k, v in [("not specific/MONDO", not r.specific), ("edge present in training HKG", not r.absent), ("no literature row", not r.has_literature)] if v))
    (out / "selection_log.md").write_text("\n".join(log) + "\n"); print("\n".join(log)); print(R[["case_type", "drug_id", "mondo_id", "score", "rank_among_new_positives", "absent_from_training_HKG", "reference_id"]].to_string(index=False))


if __name__ == "__main__":
    main()
