"""Step 8 (Comments 1.5, 2.8): audit of the RLHF preference data against the expert-evaluation data.

Answers, from the raw annotation files only (nothing is assumed):
  1. How many drug-disease pairs appear in BOTH the RLHF preference set (Section 3.6.1) and the expert-evaluation set (Section 5.4)?
  2. Which experts took part in RLHF annotation only, evaluation only, or both?
  3. How many preference comparisons were collected, and how many experts judged each comparison?
Extra checks that the reviewers' question implies: overlap at drug level and disease level, overlap of the RLHF pairs with the test
split, and, per expert who did both, how many of the evaluated pairs that expert had already seen in RLHF.

Inputs (csv or jsonl; column names are auto-detected, or pass them with --*-col)
  --rlhf   one row per expert judgement. Needs a pair key (pair_id, or drug_id + disease_id) and an expert id. Optional comparison id
           (comparison_id / comp_id); if absent, a comparison is defined as (pair, explanation_a, explanation_b) when those columns
           exist, otherwise as the pair.
  --eval   expert-evaluation ratings, one row per expert x pair (pair key + expert id), or --eval-pairs a plain list of the 200 pair ids
           (data/expert_pairs_200.txt, one "drug_id|disease_id" or pair_id per line).
  --test   optional test split csv (drug_id, disease_id) to check whether RLHF pairs were also test pairs.
Pair keys are normalised to "drug|disease" (strings, stripped, case-insensitive) in both directions of the id spelling.

Usage
  python -m exdrkg_rerun.human_eval.rlhf_audit --root . --rlhf data/rlhf/preferences.csv --eval results/expert/ratings.csv \\
        [--eval-pairs data/expert_pairs_200.txt] [--test data/splits/standard_random/test.csv]
Outputs (results/rerun/audit/): R-I_part_a_summary.json, R-I_part_a.md (text ready for the response letter), overlap_pairs.csv, experts_roles.csv,
comparisons_per_expert_count.csv, comparison_table.csv, rlhf_pairs.csv, eval_pairs.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd

ALIASES = {"pair": ["pair_id", "pair", "pair_key"], "drug": ["drug_id", "drug", "head"], "dis": ["disease_id", "disease", "tail"],
           "expert": ["expert_id", "annotator_id", "annotator", "rater_id", "rater", "expert"], "comp": ["comparison_id", "comp_id", "comparison", "task_id"],
           "a": ["explanation_a", "response_a", "option_a", "a", "chosen"], "b": ["explanation_b", "response_b", "option_b", "b", "rejected"]}


def read_any(p):
    p = Path(p)
    if p.suffix in (".jsonl", ".json"): return pd.read_json(p, lines=p.suffix == ".jsonl")
    return pd.read_csv(p)


def pick(df, key, given=None):
    if given: return given if given in df.columns else (_ for _ in ()).throw(SystemExit(f"column {given} not in {list(df.columns)}"))
    for c in ALIASES[key]:
        if c in df.columns: return c
    return None


def norm(x): return str(x).strip().lower()


def pair_key(df, a):
    pc = pick(df, "pair", getattr(a, "pair_col", None)); dc, sc = pick(df, "drug"), pick(df, "dis")
    if dc and sc: return (df[dc].map(norm) + "|" + df[sc].map(norm))
    if pc: return df[pc].map(norm).str.replace(r"\s*[|,;:/]\s*", "|", regex=True)
    raise SystemExit(f"cannot build a pair key from columns {list(df.columns)}; give drug_id/disease_id or pair_id")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--rlhf", required=True); ap.add_argument("--eval", default=None)
    ap.add_argument("--eval-pairs", default=None); ap.add_argument("--test", default=None); ap.add_argument("--out", default="results/rerun/audit")
    ap.add_argument("--rlhf-expert-col", default=None); ap.add_argument("--eval-expert-col", default=None); ap.add_argument("--comparison-col", default=None); ap.add_argument("--choice-col", default=None, help="column with the expert preference (A/B, 0/1 or the chosen text); needed for the agreement row of Table R-I(a)")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    if not (a.eval or a.eval_pairs): raise SystemExit("give --eval (ratings) and/or --eval-pairs (list of pair ids)")

    R = read_any(root / a.rlhf); R["pair"] = pair_key(R, a); ec = pick(R, "expert", a.rlhf_expert_col)
    if ec is None: raise SystemExit(f"no expert column in RLHF file: {list(R.columns)}")
    R["expert"] = R[ec].map(norm); n_rows = len(R)
    cc, ca, cb = pick(R, "comp", a.comparison_col), pick(R, "a"), pick(R, "b")
    if cc: R["comparison"] = R[cc].map(norm); comp_def = f"column '{cc}'"
    elif ca and cb and ca != cb: R["comparison"] = R.pair + "||" + R[ca].astype(str).str.strip() + "||" + R[cb].astype(str).str.strip(); comp_def = f"(pair, '{ca}', '{cb}')"
    else: R["comparison"] = R.pair; comp_def = "pair (no comparison id or A/B columns found: one comparison per pair assumed - CHECK)"

    E, eval_pairs, eval_experts = None, set(), set()
    if a.eval:
        E = read_any(root / a.eval); E["pair"] = pair_key(E, a); ee = pick(E, "expert", a.eval_expert_col)
        if ee is None: raise SystemExit(f"no expert column in eval file: {list(E.columns)}")
        E["expert"] = E[ee].map(norm); eval_pairs |= set(E.pair); eval_experts = set(E.expert)
    if a.eval_pairs:
        lst = {norm(l).replace(",", "|").replace(";", "|") for l in open(root / a.eval_pairs) if l.strip() and not l.startswith("#")}
        if E is not None and lst != eval_pairs: print(f"WARNING: --eval-pairs ({len(lst)}) differs from pairs in --eval ({len(eval_pairs)}); using the union")
        eval_pairs |= lst
    rlhf_pairs = set(R.pair)

    # 1. pair overlap (exact, drug level, disease level)
    ov = sorted(rlhf_pairs & eval_pairs)
    dr = lambda s: {p.split("|")[0] for p in s}; di = lambda s: {p.split("|")[1] for p in s if "|" in p}
    ovd = dr(rlhf_pairs) & dr(eval_pairs); ovs = di(rlhf_pairs) & di(eval_pairs)
    pd.DataFrame({"pair": ov}).to_csv(out / "overlap_pairs.csv", index=False)
    pd.DataFrame({"pair": sorted(rlhf_pairs)}).to_csv(out / "rlhf_pairs.csv", index=False); pd.DataFrame({"pair": sorted(eval_pairs)}).to_csv(out / "eval_pairs.csv", index=False)

    # 2. expert roles
    rl_ex = set(R.expert); allx = sorted(rl_ex | eval_experts) if eval_experts else sorted(rl_ex)
    roles = pd.DataFrame({"expert": allx, "RLHF_annotation": [x in rl_ex for x in allx], "final_evaluation": [x in eval_experts for x in allx]})
    roles["role"] = roles.apply(lambda r: "both" if r.RLHF_annotation and r.final_evaluation else "RLHF only" if r.RLHF_annotation else "evaluation only", axis=1)
    rj = R.groupby("expert").agg(rlhf_judgements=("comparison", "size"), rlhf_pairs=("pair", "nunique")); roles = roles.merge(rj, left_on="expert", right_index=True, how="left")
    if E is not None:
        ej = E.groupby("expert").agg(eval_pairs_rated=("pair", "nunique")); roles = roles.merge(ej, left_on="expert", right_index=True, how="left")
        seen = {x: len(set(E[E.expert == x].pair) & set(R[R.expert == x].pair)) for x in allx}; roles["eval_pairs_that_same_expert_saw_in_RLHF"] = roles.expert.map(seen)
    roles.to_csv(out / "experts_roles.csv", index=False)

    # 3. comparisons and judges per comparison
    per = R.groupby("comparison").agg(n_experts=("expert", "nunique"), n_judgements=("expert", "size"), pair=("pair", "first")); per["duplicate_judgements_same_expert"] = per.n_judgements - per.n_experts
    per.to_csv(out / "comparison_table.csv"); dist = per.n_experts.value_counts().sort_index().rename_axis("experts_per_comparison").reset_index(name="comparisons")
    dist.to_csv(out / "comparisons_per_expert_count.csv", index=False)

    # agreement on preference labels (plan R-I part a)
    agree = None; chc = a.choice_col or next((c for c in ("preference", "preferred", "choice", "winner", "chosen_label", "label") if c in R.columns and c not in ("comparison",)), None)
    if chc is not None:
        R["_choice"] = R[chc].astype(str).str.strip().str.lower(); pw = []; unan = []
        for _, g in R.groupby("comparison"):
            c = g["_choice"].tolist()
            if len(c) >= 2: pairs_ = [(x, y) for i, x in enumerate(c) for y in c[i + 1:]]; pw.append(sum(x == y for x, y in pairs_) / len(pairs_)); unan.append(len(set(c)) == 1)
        if pw: agree = {"choice_column": chc, "comparisons_with_2plus_judges": len(pw), "mean_pairwise_agreement_pct": round(100 * sum(pw) / len(pw), 1), "unanimous_comparisons_k": int(sum(unan)), "unanimous_comparisons_pct": round(100 * sum(unan) / len(unan), 1)}
    tst = None
    if a.test:
        T = read_any(root / a.test); T["pair"] = pair_key(T, a); tst = {"test_pairs": int(T.pair.nunique()), "rlhf_pairs_also_in_test": len(rlhf_pairs & set(T.pair)), "eval_pairs_also_in_test": len(eval_pairs & set(T.pair))}

    S = {"rlhf_rows": n_rows, "rlhf_unique_pairs": len(rlhf_pairs), "eval_unique_pairs": len(eval_pairs), "overlapping_pairs": len(ov),
         "overlap_pct_of_eval": round(100 * len(ov) / max(1, len(eval_pairs)), 1), "overlap_pct_of_rlhf": round(100 * len(ov) / max(1, len(rlhf_pairs)), 1),
         "shared_drugs": len(ovd), "shared_diseases": len(ovs), "n_rlhf_experts": len(rl_ex), "n_eval_experts": len(eval_experts),
         "experts_both": int((roles.role == "both").sum()), "experts_rlhf_only": int((roles.role == "RLHF only").sum()), "experts_eval_only": int((roles.role == "evaluation only").sum()),
         "comparison_definition": comp_def, "n_comparisons": int(len(per)), "total_judgements": int(per.n_judgements.sum()),
         "experts_per_comparison": {int(k): int(v) for k, v in per.n_experts.value_counts().sort_index().items()}, "mean_experts_per_comparison": round(float(per.n_experts.mean()), 2),
         "comparisons_with_repeat_judgement_by_same_expert": int((per.duplicate_judgements_same_expert > 0).sum()), "preference_agreement": agree if agree else "not computed: no preference column (pass --choice-col)", "test_split_check": tst}
    json.dump(S, open(out / "R-I_part_a_summary.json", "w"), indent=1)
    verdict = ("No pair appears in both sets." if not ov else f"{len(ov)} pair(s) appear in both sets ({S['overlap_pct_of_eval']}% of the evaluation set): the evaluation is NOT independent of the reward-model data for those pairs; "
               "report all expert results with and without them (overlap_pairs.csv lists them) and run the independent evaluation (human_eval.rating_packets).")
    md = [f"# Step 8 audit: RLHF preference data vs expert evaluation", "",
          f"- RLHF: {n_rows} judgements, {len(rlhf_pairs)} unique pairs, {S['n_comparisons']} comparisons (defined by {comp_def}); {len(rl_ex)} experts.",
          f"- Evaluation: {len(eval_pairs)} unique pairs" + (f", {len(eval_experts)} experts." if eval_experts else " (ratings file not given, experts unknown)."),
          f"- Pair overlap: {len(ov)} ({S['overlap_pct_of_eval']}% of evaluation pairs). Shared drugs: {len(ovd)}, shared diseases: {len(ovs)}. {verdict}",
          f"- Experts: {S['experts_both']} in both, {S['experts_rlhf_only']} RLHF only, {S['experts_eval_only']} evaluation only.",
          f"- Judges per comparison: {S['experts_per_comparison']} (mean {S['mean_experts_per_comparison']}); {S['comparisons_with_repeat_judgement_by_same_expert']} comparisons have repeat judgements by one expert."]
    md.append(f"- Agreement on preference labels: {S['preference_agreement']}")
    if tst: md.append(f"- Test split: {tst['rlhf_pairs_also_in_test']} RLHF pairs and {tst['eval_pairs_also_in_test']} evaluation pairs are also test pairs.")
    md += ["", roles.to_markdown(index=False), "", dist.to_markdown(index=False)]
    (out / "R-I_part_a.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    if len(rl_ex) != 10 or (eval_experts and len(eval_experts) != 10): print("NOTE: the manuscript says a panel of 10 experts in both places; counts above differ, correct the text or the data.")


if __name__ == "__main__":
    main()
