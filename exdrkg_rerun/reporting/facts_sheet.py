"""Step 13 (facts sheet, Table R-K parts a and b): assemble everything that can be filled from files, and leave the rest marked TO FILL.

Auto-filled (from the outputs of the other steps; nothing is invented)
  * commit hash, file hashes, package versions   <- results/rerun/R-K_freeze_<split>.json (data.masked_kg)
  * GPU / CUDA / PyTorch                          <- results/rerun/gpu_info.csv, the freeze json, results/rerun/pip_freeze.txt
  * rating questions and 1/3/5 anchors            <- results/rerun/expert12/rubric.md
  * allocation, materials, blinding, tie rule     <- results/rerun/expert12/protocol.json and the analysis rule of rating_analysis
  * expert roles (RLHF / evaluation / both), dimensions rated <- results/rerun/audit/experts_roles.csv, results/rerun/expert12/assignment.csv
  * GPT-4+RAG configuration                       <- data/rag_config.json
  * Eq. 11 implementation facts                   <- what src/models/evidence_retriever.py does (recorded below as CODE FACT; confirm and reword for the paper)
Everything else (specialty, years of experience, selection criteria, institution type, harmonisation tables, conflict rule) needs YOUR records and is written as "TO FILL".
Usage  python -m exdrkg_rerun.reporting.facts_sheet --root . --split standard
Outputs results/rerun/facts/R-K_part_a.csv, R-K_part_b.csv, R-K.md
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

FILL = "TO FILL"
EQ11 = ("CODE FACT (src/models/evidence_retriever.py): paths are scored by an MLP (Linear-ReLU-Linear-ReLU-Linear) over the concatenation "
        "[mean node embedding of the path (d_model), edge-type diversity (one-hot counts over relation types), pathway coverage, semantic coherence (cosine of consecutive node embeddings)]. "
        "There is no product of learned edge weights w_r and no length penalty exp(-beta(n-2)) as written in Eq. 11; the z_dq (2d) query embedding is not used in that scorer. "
        "Either change Eq. 11 to this scorer or change the code; answer 'w_r: not used' for the reviewer unless the code is changed.")


def rubric_part(md, head):
    m = re.search(rf"## {head}.*?\n(.*?)(?=\n## |\Z)", md, re.S); return " ".join(m.group(0).split()) if m else FILL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--out", default="results/rerun/facts"); a = ap.parse_args()
    root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True); R = root / "results/rerun"
    def js(p):
        p = root / p; return json.load(open(p)) if p.exists() else {}
    fz = js(f"results/rerun/R-K_freeze_{a.split}.json"); prot = js("results/rerun/expert12/protocol.json"); rag = js("data/rag_config.json")
    rubric = (R / "expert12/rubric.md").read_text() if (R / "expert12/rubric.md").exists() else ""
    gpu = (R / "gpu_info.csv").read_text().replace("\n", " ; ") if (R / "gpu_info.csv").exists() else FILL
    ver = {k: v for k, v in fz.items() if re.search(r"torch|cuda|python|version|numpy|pyg|geometric", k, re.I)}
    hashes = {k: v for k, v in fz.items() if re.search(r"md5|hash|sha|commit", k, re.I)}
    roles = pd.read_csv(R / "audit/experts_roles.csv") if (R / "audit/experts_roles.csv").exists() else pd.DataFrame(); asg = pd.read_csv(R / "expert12/assignment.csv") if (R / "expert12/assignment.csv").exists() else pd.DataFrame()
    A = []
    for i in range(1, 11):
        rid = roles.expert.iloc[i - 1] if len(roles) >= i else None
        A.append({"Code": f"E{i}", "Role (researcher / clinician)": FILL, "Specialty or field": FILL, "Years of experience": FILL, "Background": FILL, "Institution type (anonymised)": FILL,
                  "RLHF / evaluation / both": (roles.role.iloc[i - 1] if rid is not None else FILL), "Dimensions rated": (",".join(sorted(set(asg[asg.rater_id.astype(str) == str(rid)].dimensions))) if rid is not None and len(asg) else FILL)})
    Pa = pd.DataFrame(A); Pa.to_csv(out / "R-K_part_a.csv", index=False)
    B = [("Selection criteria for experts", FILL),
         ("Exact rating question and 1 / 3 / 5 anchors for factual correctness", rubric_part(rubric, "Factual correctness")), ("Exact rating question and 1 / 3 / 5 anchors for mechanistic coherence", rubric_part(rubric, "Mechanistic coherence")),
         ("Exact rating question and 1 / 3 / 5 anchors for clinical usefulness", rubric_part(rubric, "Clinical usefulness")), ("How the 200 pairs were selected (random, stratified, score range)", FILL + " (original round; the new 100-pair set: see expert12/eval_sets.json)"),
         ("How pairs were allocated to raters", prot.get("pool_mode", FILL) + (f"; {prot['raters_per_item']} raters per item, load {prot.get('load_per_rater')}" if prot else "")),
         ("Materials available during rating", (f"evidence paths shown: {prot.get('evidence_paths_shown')}; literature access: {prot.get('literature_access')}" if prot else FILL)),
         ("What was concealed under double-blind", ("model identity (A/B/C(/D), independent random order per pair and rater), model scores, panel identity; raters do not see each other's ratings" if prot else FILL)),
         ("How ties and disagreements were handled", "item = one explanation x one dimension; if independent ratings differ by >= 2 points a senior rater adjudicates and the senior score is final; otherwise the mean of the raters (rating_analysis.py)"),
         ("Were free-text comments collected in the original round? (Yes / No)", FILL),
         ("Edge weight w_r in Eq. 11", EQ11), ("Dimensions of z_p and z_dq in Eq. 11", "z_p: d_model (mean-pooled node embeddings); the scorer input is [z_p, diversity, coverage, coherence]; z_dq is not used (see the CODE FACT above)"),
         ("Entity harmonisation method (identifier mapping tables used)", FILL + " (see results/rerun/stats/R-E3_identifier_systems.csv for the identifier systems found)"), ("Conflict resolution rule for contradictory edges", FILL + " (see R-E3_conflicts.csv; the builder does not log a rule)"),
         ("GPT-4+RAG: corpus size, chunk length, embedding model, index type, top-k, reranker", json.dumps(rag) if rag else FILL), ("Code commit hash and data file hashes (Step 1)", json.dumps(hashes) if hashes else FILL),
         ("GPU, CUDA, PyTorch versions", gpu + (" ; " + json.dumps(ver) if ver else ""))]
    Pb = pd.DataFrame(B, columns=["Item", "Value"]); Pb.to_csv(out / "R-K_part_b.csv", index=False)
    (out / "R-K.md").write_text("## Table R-K (part a): expert panel profile\n" + Pa.to_markdown(index=False) + "\n\n## Table R-K (part b): protocol and implementation facts\n" + Pb.to_markdown(index=False) + f"\n\nItems marked {FILL} need your own records.\n")
    print(f"{sum(FILL in str(v) for v in Pb.Value)} of {len(Pb)} part-b items and {int((Pa == FILL).sum().sum())} part-a cells still need your input -> {out / 'R-K.md'}")


if __name__ == "__main__":
    main()
