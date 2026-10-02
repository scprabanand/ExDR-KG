"""Step 12 analysis: Krippendorff's alpha per dimension, adjudication, per-model scores, free-text themes (Tables R-I b, R-J, R-J b).

Subcommands
  collect     read the filled rater sheets (results/rerun/expert12/rater_sheets_filled/*.csv, same format as the blind sheets), validate them (whole numbers 1-5,
              no missing cells for the assigned dimensions), un-blind with blinding_key.csv and write ratings_long.csv + comments_long.csv.
              Then it lists the items needing adjudication (rule: for one item = one explanation and one dimension, max - min of the independent ratings >= 2)
              in adjudication_needed.csv, and writes comments_to_code.csv with a keyword-based SUGGESTED theme column (a human coder must confirm/overwrite
              the `themes` column; suggestions are not results).
  analyse     needs ratings_long.csv, optionally adjudication_done.csv (pair_id,model,dimension,senior_rater,score) and comments_coded.csv (pair_id,rater_id,
              model,comment,themes; several themes separated by ';').
Reported
  * Krippendorff's alpha (ordinal) per dimension over ALL items (rows = pair x model, columns = raters, missing where a rater did not rate), with a 95%
    bootstrap CI over items (1,000, seed 42); also per model.
  * number and share of items that needed adjudication (per dimension and in total); the FINAL item score is the senior rater's score for adjudicated
    items, otherwise the mean of the independent ratings. Unadjusted means are reported next to it.
  * per model and dimension: mean +/- SD of item scores, % of items with score >= 4, paired Wilcoxon tests between models with Holm correction.
  * split by whether the rater took part in RLHF (if the roster is given), so any effect of reusing RLHF raters is visible.
  * theme counts per model (count of comments carrying a theme and % of that model's comments).
Usage
  python -m exdrkg_rerun.human_eval.rating_analysis collect --root .
  python -m exdrkg_rerun.human_eval.rating_analysis analyse --root . [--adjudication results/rerun/expert12/adjudication_done.csv] [--comments results/rerun/expert12/comments_coded.csv] [--roster data/raters.csv]
"""
import argparse, itertools, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from exdrkg_rerun.common.stats import krippendorff_alpha_ordinal

DIMS = ["FC", "MC", "CU"]
THEME_RULES = [("hallucinated_entity", r"halluc|invent|non-?existent|fabricat|unsupported entit|not in the (kg|graph)"), ("wrong_target", r"wrong target|incorrect target|does not (act|bind|target)|not a target"),
               ("missing_step", r"missing (step|link|intermediate)|skips|gap in|no intermediate|does not explain how"), ("vague_claim", r"vague|generic|non-?specific|too broad|lacks detail|unspecific"),
               ("clinically_impractical", r"impractical|dose|toxicity|safety|route|not feasible|unrealistic|side effect"), ("contradicts_evidence", r"contradict|inconsistent with|known to be false|incorrect"),
               ("correct_mechanism", r"correct mechanism|accurate|plausible chain|well[- ]explained|coherent"), ("well_supported", r"well[- ]supported|specific|cites|supported by")]


def collect(a):
    root = Path(a.root); d = root / a.out; key = pd.read_csv(d / "blinding_key.csv"); A = pd.read_csv(d / "assignment.csv"); rat, com, problems = [], [], []
    for f in sorted((d / "rater_sheets_filled").glob("*.csv")):
        r = f.stem; S = pd.read_csv(f, dtype={"item_id": str}); dims = A[A.rater_id.astype(str) == r].dimensions.iloc[0].split(",")
        for _, row in S.iterrows():
            for lab in sorted(key.label.unique()):
                k = key[(key.item_id == row.item_id) & (key.label == lab)]
                if k.empty: continue
                for dm in dims:
                    v = row.get(f"{lab}_{dm}")
                    if pd.isna(v) or str(v).strip() == "": problems.append((r, row.item_id, lab, dm, "empty")); continue
                    if float(v) != int(float(v)) or not 1 <= int(float(v)) <= 5: problems.append((r, row.item_id, lab, dm, f"bad score {v}")); continue
                    rat.append({"pair_id": k.pair_id.iloc[0], "model": k.model.iloc[0], "rater_id": r, "dimension": dm, "score": int(float(v))})
                c = row.get(f"{lab}_comment"); com.append({"pair_id": k.pair_id.iloc[0], "rater_id": r, "model": k.model.iloc[0], "comment": "" if pd.isna(c) else str(c)})
    if problems: pd.DataFrame(problems, columns=["rater", "item", "label", "dim", "problem"]).to_csv(d / "validation_problems.csv", index=False); print(f"WARNING: {len(problems)} problems, see validation_problems.csv")
    L = pd.DataFrame(rat); L.to_csv(d / "ratings_long.csv", index=False); C = pd.DataFrame(com); C.to_csv(d / "comments_long.csv", index=False)
    C["suggested_theme"] = C.comment.map(lambda t: ";".join(n for n, rx in THEME_RULES if re.search(rx, t, re.I))); C["themes"] = ""; C.to_csv(d / "comments_to_code.csv", index=False)
    g = L.groupby(["pair_id", "model", "dimension"]).score.agg(["min", "max", "count", list]).reset_index(); need = g[(g["max"] - g["min"] >= 2)]
    need.rename(columns={"list": "ratings"}).assign(senior_rater="", score="").to_csv(d / "adjudication_needed.csv", index=False)
    print(f"items (pair x model x dimension) {len(g)}; needing adjudication {len(need)} ({100 * len(need) / max(1, len(g)):.1f}%); per dimension: {need.dimension.value_counts().to_dict()}")
    print("fewer than 3 independent ratings:", int((g['count'] < 3).sum()), "items")


def alpha_ci(M, n=1000, seed=42):
    rng = np.random.default_rng(seed); est = krippendorff_alpha_ordinal(M); bs = []
    for _ in range(n):
        b = krippendorff_alpha_ordinal(M[rng.integers(0, len(M), len(M))])
        if not np.isnan(b): bs.append(b)
    return est, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"),) * 2


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj


def analyse(a):
    root = Path(a.root); d = root / a.out; L = pd.read_csv(d / "ratings_long.csv"); res = {}; models = sorted(L.model.unique())
    rows = []
    for dm in DIMS:
        g = L[L.dimension == dm]
        if g.empty: continue
        for scope in ["all"] + models:
            h = g if scope == "all" else g[g.model == scope]; P = h.assign(item=h.pair_id + "|" + h.model).pivot_table(index="item", columns="rater_id", values="score", aggfunc="first")
            est, ci = alpha_ci(P.values.astype(float)); rows.append({"dimension": dm, "scope": scope, "items": len(P), "raters": P.shape[1], "alpha_ordinal": round(est, 3), "ci_low": round(ci[0], 3), "ci_high": round(ci[1], 3)})
    Al = pd.DataFrame(rows); Al.to_csv(d / "R-I_alpha.csv", index=False)
    # adjudication
    g = L.groupby(["pair_id", "model", "dimension"]).score.agg(["min", "max", "mean", "count"]).reset_index(); g["needs_adj"] = (g["max"] - g["min"] >= 2)
    adj = pd.read_csv(root / a.adjudication) if a.adjudication and (root / a.adjudication).exists() else pd.DataFrame(columns=["pair_id", "model", "dimension", "score"])
    g = g.merge(adj[["pair_id", "model", "dimension", "score"]].rename(columns={"score": "senior"}), on=["pair_id", "model", "dimension"], how="left")
    missing = int((g.needs_adj & g.senior.isna()).sum()); g["final"] = np.where(g.needs_adj & g.senior.notna(), g.senior, g["mean"])
    n_adj = g.groupby("dimension").needs_adj.agg(["sum", "count"]); n_adj.loc["ALL"] = n_adj.sum(); n_adj["pct"] = (100 * n_adj["sum"] / n_adj["count"]).round(1); n_adj.to_csv(d / "R-I_adjudication_counts.csv")
    S = g.groupby(["model", "dimension"]).agg(mean_unadjusted=("mean", "mean"), mean_final=("final", "mean"), sd_final=("final", "std"), pct_final_ge4=("final", lambda x: 100 * (x >= 4).mean()), items=("final", "size")).round(3).reset_index()
    S.to_csv(d / "R-I_scores.csv", index=False)
    tests = []
    for dm in DIMS:
        piv = g[g.dimension == dm].pivot_table(index="pair_id", columns="model", values="final")
        pr = list(itertools.combinations(piv.columns, 2)); ps = []
        for x, y in pr:
            z = piv[[x, y]].dropna(); diff = z[x] - z[y]; ps.append(1.0 if (diff == 0).all() else wilcoxon(z[x], z[y]).pvalue)
        if ps:
            for (x, y), p, q in zip(pr, ps, holm(np.array(ps))): tests.append({"dimension": dm, "a": x, "b": y, "p": p, "p_holm": q, "significant_0.05": bool(q < 0.05)})
    pd.DataFrame(tests).to_csv(d / "R-I_pairwise_wilcoxon.csv", index=False)
    if a.roster and (root / a.roster).exists():
        R = pd.read_csv(root / a.roster).astype({"rater_id": str}); X = L.astype({"rater_id": str}).merge(R[["rater_id", "took_part_in_rlhf"]], on="rater_id")
        X.groupby(["took_part_in_rlhf", "model", "dimension"]).score.agg(["mean", "std", "count"]).round(3).to_csv(d / "R-I_by_rater_rlhf_status.csv")
    themes = None
    if a.comments and (root / a.comments).exists():
        C = pd.read_csv(root / a.comments).fillna(""); C = C[C.themes.astype(str).str.strip() != ""]
        T = C.assign(t=C.themes.str.split(";")).explode("t"); T["t"] = T.t.str.strip(); T = T[T.t != ""]
        tot = C.groupby("model").size(); themes = T.groupby(["t", "model"]).size().unstack(fill_value=0); pct = (themes / tot * 100).round(1); themes.to_csv(d / "R-J_b_theme_counts.csv"); pct.to_csv(d / "R-J_b_theme_percent_of_comments.csv")
    md = ["# Step 12 results", "", "## Krippendorff's alpha (ordinal)", Al.to_markdown(index=False), "", "## Adjudication (>= 2-point spread)", n_adj.to_markdown(),
          f"({missing} items still need a senior score; their final score currently falls back to the mean)" if missing else "", "", "## Scores per model", S.to_markdown(index=False), "",
          "## Pairwise Wilcoxon (Holm)", pd.DataFrame(tests).round(4).to_markdown(index=False) if tests else "n/a"]
    if themes is not None: md += ["", "## Theme counts per model", themes.to_markdown()]
    if themes is not None:
        PLAN = {"Correct and specific mechanism": ["correct_mechanism", "well_supported"], "Missing intermediate step": ["missing_step"], "Wrong or hallucinated target": ["wrong_target", "hallucinated_entity"],
                "Vague or generic claim": ["vague_claim"], "Clinically impractical suggestion": ["clinically_impractical"]}
        rowsj = []
        for lab, codes in PLAN.items():
            r_ = {"Theme from free-text comments": lab}
            for m in themes.columns: r_[f"{m} (count)"] = int(sum(themes.loc[c, m] for c in codes if c in themes.index))
            q = T[T.t.isin(codes)]; q = q[q.comment.astype(str).str.len().between(15, 160)]
            r_["Example comment (short quote)"] = q.sort_values("comment", key=lambda x: x.str.len()).comment.iloc[0] if len(q) else ""
            rowsj.append(r_)
        pd.DataFrame(rowsj).to_csv(d / "R-J_b_qualitative_feedback.csv", index=False); (d / "R-J_b_qualitative_feedback.md").write_text(pd.DataFrame(rowsj).to_markdown(index=False))
    (d / "R-I_R-J_summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("collect", "analyse"): p = sub.add_parser(n); p.add_argument("--root", default="."); p.add_argument("--out", default="results/rerun/expert12")
    q = sub.choices["analyse"]; q.add_argument("--adjudication", default=None); q.add_argument("--comments", default=None); q.add_argument("--roster", default=None)
    a = ap.parse_args(); collect(a) if a.cmd == "collect" else analyse(a)


if __name__ == "__main__":
    main()
