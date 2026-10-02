"""Step 12 (Comments 1.5, 2.2-2.4, 2.8): independent blinded expert evaluation - sampling, rater assignment and rating packets.

Subcommands
  sample    draw 100 NEW test pairs (never used in RLHF, not among the earlier 200 expert pairs), stratified by disease family, seed 42.
            Writes step12_pairs.txt and Table R-J (family: available / sampled). Run BEFORE generating the explanations, so the same pairs can be
            explained by every model (generate_explanations.py reads results/rerun/expert12/eval_sets.json like the Step 4 sets; see README).
  packets   needs explanations for those pairs from the 3 systems (jsonl {"pair_id","model","explanation"}), the rater roster, the rubric flags.
            Builds, per rater, a BLIND sheet (models shown as A/B/C in an independent random order per pair and rater, no model names, scores or panel
            identity), the blinding key (kept separately), the assignment table, the written rubric with anchors 1/3/5, the codebook of comment themes,
            protocol.json (states evidence-path and literature access, RLHF overlap of raters) and a leak check (model names inside explanations).
Design rules (all recorded in protocol.json)
  * every explanation is rated by >= 3 raters INDEPENDENTLY (--raters-per-item, default 3); load is balanced across raters;
  * raters who took part in RLHF annotation are used only if there are too few others (or --allow-rlhf-raters); this is flagged in the assignment log
    and must be stated in the paper;
  * pool mode `split` (default): pharmacology / molecular-biology raters rate factual correctness (FC) and mechanistic coherence (MC), clinicians rate
    clinical usefulness (CU); `--all-dimensions`: every rater rates all three. Who rated what is in assignment.csv;
  * senior raters (role = senior) never appear in the first-round assignment; they adjudicate items with a >= 2-point spread (rating_analysis.py).
Inputs
  --test        test split csv (drug_id, disease_id, label)
  --family-map  csv disease_id,family   (a real disease-family map is required for stratification; nothing is invented)
  --exclude     text files of pair keys "drug_id|disease_id" to exclude (RLHF preference pairs, earlier 200 expert pairs), one or more
  --roster      csv rater_id,role(pharm|clinician|senior),took_part_in_rlhf(0/1)
  --evidence    optional evidence.jsonl (pair_id, drug, disease, paths); drug/disease names come from it (or --names json {id: name})
Usage
  python -m exdrkg_rerun.human_eval.rating_packets sample --root . --test data/splits/standard_degree_matched/test.csv --family-map data/disease_family.csv --exclude data/rlhf/pairs.txt data/expert_pairs_200.txt --pos-frac 0.5
  python -m exdrkg_rerun.human_eval.rating_packets packets --root . --explanations results/rerun/expert12/explanations.jsonl --models ExDR-KG "GPT-4o+RAG" BioGPT --roster data/raters.csv --show-paths no --literature-access no
"""
import argparse, hashlib, json, random, re
from pathlib import Path
import numpy as np, pandas as pd

RUBRIC = """# Rating rubric (identical for every rater)
You will see, for each drug-disease pair, three explanations labelled A, B and C (order is random; the source is not disclosed). Rate each explanation
independently on the dimensions assigned to you, using whole numbers 1 to 5. Then write ONE free-text comment per explanation.

## Factual correctness (FC) - are the stated biomedical facts true?
1 = mostly false or invented entities/interactions (e.g. a named target that the drug does not act on).
3 = mixed: the main claim is plausible but at least one stated fact is wrong, unverifiable, or too vague to check.
5 = every stated fact (targets, pathways, relations) is correct and checkable in the literature.
## Mechanistic coherence (MC) - does the causal chain from drug to disease hold together?
1 = no coherent chain; steps are unrelated or contradictory.
3 = a chain is present but a key intermediate step is missing or a link is asserted without a plausible biological basis.
5 = complete, ordered chain drug -> target -> pathway/process -> disease with each link biologically plausible.
## Clinical usefulness (CU) - would this help a clinician or researcher decide whether to pursue the repurposing hypothesis?
1 = not usable: misleading, or ignores route/dose/safety/population realities.
3 = partly useful: gives a starting hypothesis but leaves practical questions (evidence level, population, feasibility) unaddressed.
5 = directly useful: specific, testable, and clinically realistic.
## Free-text comment (one per explanation)
Note strengths, mechanistic errors, hallucinated or unsupported entities, and practical limitations. If you use any of the codes in codebook.md you may
add them in brackets, but free wording is fine.
## Conditions
Do not try to identify the system that wrote an explanation. Rate each explanation on its own; ties are allowed. {conditions}
"""
CODEBOOK = """theme,definition
correct_mechanism,the drug-target-pathway-disease chain is correct and complete
missing_step,a plausible chain but an intermediate step is missing
wrong_target,a target/pathway/gene is named that the drug does not act on
hallucinated_entity,an entity that does not exist or is unsupported by the evidence
vague_claim,generic statement without a specific mechanism
clinically_impractical,mechanism plausible but unrealistic clinically (dose, route, safety, population)
contradicts_evidence,statement conflicts with known literature
well_supported,claims are specific and supported by cited evidence or paths
other,anything else (state in the comment)
"""
LEAK = re.compile(r"(gpt-?\d|openai|biomistral|biogpt|llama|mistral|txgnn|exdr|claude|gemini|as an ai|language model)", re.I)


def sample(a):
    root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True); T = pd.read_csv(root / a.test)
    T["pair"] = T.drug_id.astype(str) + "|" + T.disease_id.astype(str); T = T.drop_duplicates("pair")
    F = pd.read_csv(root / a.family_map).astype(str); T = T.merge(F[["disease_id", "family"]].assign(disease_id=lambda d: d.disease_id.astype(str)), on="disease_id", how="left")
    if T.family.isna().any(): print(f"WARNING: {int(T.family.isna().sum())} test pairs have no family; they are put in 'unmapped' and cannot be the basis of the stratification claim"); T["family"] = T.family.fillna("unmapped")
    ex = set(); [ex.update(l.strip() for l in open(root / p) if l.strip()) for p in a.exclude]; pool = T[~T.pair.isin(ex)].copy()
    print(f"test pairs {len(T)}, excluded {len(T) - len(pool)} (RLHF / earlier expert pairs), pool {len(pool)}")
    n_pos = round(a.n * a.pos_frac); rnd = random.Random(a.seed); chosen = []
    for lab, n in ((1, n_pos), (0, a.n - n_pos)):
        sub = pool[pool.label == lab]; n = min(n, len(sub))
        if n < (n_pos if lab else a.n - n_pos): print(f"WARNING: only {len(sub)} available with label {lab}")
        cnt = sub.family.value_counts(); quota = (cnt / cnt.sum() * n); base = np.floor(quota).astype(int); base = base.clip(lower=0)
        base[base == 0] = 1 if n >= len(cnt) else 0                     # every family represented when possible
        while base.sum() > n: base[base.idxmax()] -= 1
        rem = (quota - np.floor(quota)).sort_values(ascending=False)
        for fam in rem.index:
            if base.sum() >= n: break
            if base[fam] < cnt[fam]: base[fam] += 1
        for fam, k in base.items():
            g = sub[sub.family == fam].pair.tolist(); rnd.shuffle(g); chosen += g[:min(k, len(g))]
    S = pool[pool.pair.isin(chosen)]; comp = pd.DataFrame({"family": pool.family.value_counts().index}).assign(available=lambda d: d.family.map(pool.family.value_counts()),
             sampled=lambda d: d.family.map(S.family.value_counts()).fillna(0).astype(int), sampled_positives=lambda d: d.family.map(S[S.label == 1].family.value_counts()).fillna(0).astype(int))
    if a.old_set and (root / a.old_set).exists():
        old = {l.strip() for l in open(root / a.old_set) if l.strip()}; O = T[T.pair.isin(old)]; comp["pairs_in_200_set"] = comp.family.map(O.family.value_counts()).fillna(0).astype(int)
        comp = comp.rename(columns={"sampled": "pairs_in_new_100_set"}); comp.loc[len(comp)] = ["Total"] + [int(comp[c].sum()) if comp[c].dtype != object else "" for c in comp.columns[1:]]
        if len(old - set(T.pair)): print(f"NOTE: {len(old - set(T.pair))} of the 200 old pairs are not in the test file, so their family is unknown here; add them to the family map for the full composition")
    comp.to_csv(out / "R-J_composition.csv", index=False); (out / "R-J_composition.md").write_text(comp.to_markdown(index=False))
    (out / "step12_pairs.txt").write_text("\n".join(sorted(S.pair)) + "\n")
    for h in ("H1", "H2"): (out / f"{h}_pairs.txt").write_text("\n".join(sorted(S.pair)) + "\n")   # so generate_explanations.py --sets results/rerun/expert12 can be reused unchanged
    json.dump({"n": len(S), "positives": int(S.label.sum()), "seed": a.seed, "pos_frac": a.pos_frac, "excluded_files": a.exclude, "md5": hashlib.md5("\n".join(sorted(S.pair)).encode()).hexdigest(),
               "overlap_with_excluded": len(set(S.pair) & ex), "definition": f"{len(S)} new test pairs, stratified by disease family (proportional, every family >=1 where possible), seed {a.seed}"},
              open(out / "eval_sets.json", "w"), indent=1)
    print(comp.to_string(index=False)); print(f"sampled {len(S)} pairs -> {out / 'step12_pairs.txt'}. Generate the explanations for these pairs with all systems, then run 'packets'.")


def packets(a):
    root = Path(a.root); out = root / a.out; sheets = out / "rater_sheets"; sheets.mkdir(parents=True, exist_ok=True); rnd = random.Random(a.seed)
    pairs = [l.strip() for l in open(out / "step12_pairs.txt") if l.strip()]; R = pd.read_csv(root / a.roster).astype({"rater_id": str}); models = a.models
    expl = {}
    for j in map(json.loads, open(root / a.explanations)):
        if j["model"] in models and j["pair_id"] in set(pairs): expl[(j["pair_id"], j["model"])] = j["explanation"].strip()
    miss = [(p, m) for p in pairs for m in models if (p, m) not in expl]
    if miss: raise SystemExit(f"{len(miss)} (pair, model) explanations missing, e.g. {miss[:3]}; every system must explain every sampled pair")
    leaks = [(k, LEAK.search(v).group(0)) for k, v in expl.items() if LEAK.search(v)]
    if leaks: print(f"WARNING: {len(leaks)} explanations contain model-revealing text (e.g. {leaks[:3]}). Redact or normalise them before sending: blinding is otherwise broken.")
    ev = {j["pair_id"]: j for j in map(json.loads, open(root / a.evidence))} if a.evidence and (root / a.evidence).exists() else {}
    names = json.load(open(root / a.names)) if a.names else {}
    nm = lambda p, i: (ev.get(p, {}).get("drug" if i == 0 else "disease")) or names.get(p.split("|")[i], p.split("|")[i])
    senior = R[R.role == "senior"]; first = R[R.role != "senior"]; rlhf_used = False
    def pool(roles):
        nonlocal rlhf_used
        c = first[first.role.isin(roles)]; free = c[c.took_part_in_rlhf.astype(int) == 0]
        if len(free) >= a.raters_per_item and not a.allow_rlhf_raters: return free.rater_id.tolist()
        rlhf_used = rlhf_used or len(c) > len(free) or a.allow_rlhf_raters; return c.rater_id.tolist()
    if a.all_dimensions: groups = {"all": (pool(["pharm", "clinician"]), ["FC", "MC", "CU"])}
    else: groups = {"pharm": (pool(["pharm"]), ["FC", "MC"]), "clinician": (pool(["clinician"]), ["CU"])}
    for g, (p_, d_) in groups.items():
        if len(p_) < a.raters_per_item: raise SystemExit(f"pool '{g}' has {len(p_)} raters, need >= {a.raters_per_item}")
    load = {r: 0 for g in groups.values() for r in g[0]}; assign = []
    for p in pairs:
        for g, (members, dims) in groups.items():
            pick = sorted(members, key=lambda r: (load[r], rnd.random()))[:a.raters_per_item]
            for r in pick: load[r] += 1; assign.append({"pair_id": p, "rater_id": r, "group": g, "dimensions": ",".join(dims)})
    A = pd.DataFrame(assign); A.to_csv(out / "assignment.csv", index=False); key = []
    A = A.merge(R[["rater_id", "role", "took_part_in_rlhf"]], on="rater_id")
    for r, g in A.groupby("rater_id"):
        dims = g.dimensions.iloc[0].split(","); rows = []; order = g.pair_id.tolist(); rnd.shuffle(order)
        for p in order:
            perm = models[:]; rnd.shuffle(perm); item = hashlib.md5(f"{p}|{r}|{a.seed}".encode()).hexdigest()[:8]
            row = {"item_id": item, "drug": nm(p, 0), "disease": nm(p, 1)}
            if a.show_paths == "yes": row["evidence_paths"] = " || ".join(ev.get(p, {}).get("paths", [])[:10])
            for lab, m in zip("ABCD"[:len(models)], perm):
                row[f"explanation_{lab}"] = expl[(p, m)]; key.append({"item_id": item, "pair_id": p, "rater_id": r, "label": lab, "model": m})
                for d in dims: row[f"{lab}_{d}"] = ""
                row[f"{lab}_comment"] = ""
            rows.append(row)
        pd.DataFrame(rows).to_csv(sheets / f"{r}.csv", index=False)
    pd.DataFrame(key).to_csv(out / "blinding_key.csv", index=False)              # keep away from raters
    cond = (f"Retrieved evidence paths were {'shown' if a.show_paths == 'yes' else 'NOT shown'}; literature/internet access during rating was {'allowed' if a.literature_access == 'yes' else 'NOT allowed'}.")
    (out / "rubric.md").write_text(RUBRIC.format(conditions=cond)); (out / "codebook.csv").write_text(CODEBOOK)
    P = {"n_pairs": len(pairs), "models": models, "raters_per_item": a.raters_per_item, "pool_mode": "all_dimensions" if a.all_dimensions else "split (pharm: FC+MC, clinician: CU)", "evidence_paths_shown": a.show_paths,
         "literature_access": a.literature_access, "seed": a.seed, "raters_first_round": first.rater_id.tolist(), "senior_adjudicators": senior.rater_id.tolist(),
         "raters_who_took_part_in_rlhf_used": sorted(set(A[A.took_part_in_rlhf.astype(int) == 1].rater_id)), "rlhf_raters_used": bool(rlhf_used or (A.took_part_in_rlhf.astype(int) == 1).any()),
         "load_per_rater": load, "leak_warnings": len(leaks)}
    json.dump(P, open(out / "protocol.json", "w"), indent=1)
    print(json.dumps(P, indent=1)); print(f"rater sheets: {sheets} (send ONLY these + rubric.md + codebook.csv); keep blinding_key.csv and assignment.csv private")
    if P["rlhf_raters_used"]: print("NOTE: some raters took part in RLHF annotation; the paper must say so explicitly (protocol.json lists them).")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("sample", "packets"):
        p = sub.add_parser(n); p.add_argument("--root", default="."); p.add_argument("--out", default="results/rerun/expert12"); p.add_argument("--seed", type=int, default=42)
    s = sub.choices["sample"]; s.add_argument("--test", required=True); s.add_argument("--family-map", required=True); s.add_argument("--exclude", nargs="+", required=True)
    s.add_argument("--old-set", default=None, help="the earlier 200-pair set (pair keys) for the composition table R-J part a"); s.add_argument("--n", type=int, default=100); s.add_argument("--pos-frac", type=float, default=0.5)
    k = sub.choices["packets"]; k.add_argument("--explanations", required=True); k.add_argument("--models", nargs="+", required=True, help="3 or 4 systems (plan R-I b lists four: BioGPT, GPT-4+RAG, TxGNN+Template, ExDR-KG); shown as A, B, C, D"); k.add_argument("--roster", required=True)
    k.add_argument("--evidence", default="results/rerun/hallucination/evidence.jsonl"); k.add_argument("--names", default=None); k.add_argument("--raters-per-item", type=int, default=3)
    k.add_argument("--all-dimensions", action="store_true"); k.add_argument("--allow-rlhf-raters", action="store_true")
    k.add_argument("--show-paths", choices=["yes", "no"], required=True); k.add_argument("--literature-access", choices=["yes", "no"], required=True)
    a = ap.parse_args(); sample(a) if a.cmd == "sample" else packets(a)


if __name__ == "__main__":
    main()
