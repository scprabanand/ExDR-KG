"""Step 4, items 1-2: define and record the two hallucination evaluation sets.

H1  the 200 expert-evaluated pairs. Give their pair IDs with --h1-file (one "drug_id|disease_id" per line, the exact
    pairs shown to the experts). If the file is missing the script REFUSES to invent H1 unless --allow-new-h1 is passed;
    a newly drawn H1 is not the set the experts rated and must be described that way in the manuscript.
H2  the larger automatic set, defined explicitly by --h2-mode (the definition string is written to the manifest):
      test_sample   all positives of the chosen split's test file + a seeded random sample of its negatives, N in total
                    (default N = 500). The standard test file has only 115-116 positives, so "all test positives" alone
                    cannot give 500 pairs; this mode pads with negatives and says so.
      top_ranked    the N test pairs with the highest score in --score-file (a model's score CSV); useful for checking
                    explanations on the pairs a user would actually read.
      pooled        all positives of the standard test file and of the zero-shot test file (--zero-shot-split) plus a seeded
                    sample of negatives from both, N in total.
Outputs (results/rerun/hallucination/): H1_pairs.txt, H2_pairs.txt, eval_sets.json (definitions, N, composition, md5, overlap H1&H2).
Every later script reads these files, so the same pairs are used for every model.

Usage
  python -m exdrkg_rerun.explainability.build_eval_sets --root . --split standard_degree_matched --h1-file data/expert_pairs_200.txt --h2-mode test_sample --n-h2 500
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd


def pid(df): return (df.drug_id.astype(str) + "|" + df.disease_id.astype(str)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--zero-shot-split", default="zero_shot")
    ap.add_argument("--h1-file", default=None); ap.add_argument("--allow-new-h1", action="store_true"); ap.add_argument("--n-h1", type=int, default=200)
    ap.add_argument("--h2-mode", choices=["test_sample", "top_ranked", "pooled"], default="test_sample"); ap.add_argument("--n-h2", type=int, default=500)
    ap.add_argument("--score-file", default=None); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="results/rerun/hallucination")
    a = ap.parse_args(); root = Path(a.root); rng = np.random.default_rng(a.seed); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(root / "data/splits" / a.split / "test.csv"); test["pair_id"] = pid(test)
    known = set(test.pair_id)
    # ---------------- H1
    if a.h1_file:
        h1 = [l.strip() for l in open(a.h1_file) if l.strip()]
        assert len(h1) == len(set(h1)), "H1 file contains repeated pair IDs"
        h1_note = f"the {len(h1)} pairs rated by the expert panel (IDs supplied in {a.h1_file})"
        outside = [p for p in h1 if p not in known]
        if outside: h1_note += f"; WARNING: {len(outside)} of them are not in the current {a.split} test file (the expert set predates the split rebuild)"
    elif a.allow_new_h1:
        pos = test[test.label == 1]; neg = test[test.label == 0]
        n_pos = min(len(pos), a.n_h1 // 2)
        h1 = pos.sample(n_pos, random_state=a.seed).pair_id.tolist() + neg.sample(a.n_h1 - n_pos, random_state=a.seed).pair_id.tolist()
        h1_note = f"NEW draw (not the original expert-rated pairs): {n_pos} positives + {a.n_h1 - n_pos} negatives, seed {a.seed}"
    else:
        raise SystemExit("H1 needs the pair IDs of the 200 expert-evaluated pairs (--h1-file). Use --allow-new-h1 only if they are lost, and say so in the paper.")
    # ---------------- H2
    if a.h2_mode == "test_sample":
        pos = test[test.label == 1]; neg = test[test.label == 0]; k = min(len(pos), a.n_h2)
        h2 = pos.pair_id.tolist() + neg.sample(a.n_h2 - k, random_state=a.seed).pair_id.tolist()
        h2_def = f"All {len(pos)} positive pairs of the {a.split} test file plus {a.n_h2 - k} negatives sampled uniformly (seed {a.seed}); N = {len(h2)}"
    elif a.h2_mode == "top_ranked":
        if not a.score_file: raise SystemExit("--score-file is required for top_ranked")
        sc = pd.read_csv(a.score_file); sc["pair_id"] = pid(sc)
        h2 = sc.sort_values("score", ascending=False).head(a.n_h2).pair_id.tolist()
        h2_def = f"The {a.n_h2} test pairs with the highest score in {a.score_file}; N = {len(h2)}"
    else:
        zs = pd.read_csv(root / "data/splits" / a.zero_shot_split / "test.csv"); zs["pair_id"] = pid(zs)
        pool = pd.concat([test.assign(src=a.split), zs.assign(src=a.zero_shot_split)]).drop_duplicates("pair_id")
        pos = pool[pool.label == 1]; neg = pool[pool.label == 0]; k = min(len(pos), a.n_h2)
        h2 = pos.pair_id.tolist() + neg.sample(a.n_h2 - k, random_state=a.seed).pair_id.tolist()
        h2_def = (f"All positives of the {a.split} and {a.zero_shot_split} test files ({len(pos)}) plus {a.n_h2 - k} negatives sampled uniformly "
                  f"from their union (seed {a.seed}); N = {len(h2)}")
    (out / "H1_pairs.txt").write_text("\n".join(h1) + "\n"); (out / "H2_pairs.txt").write_text("\n".join(h2) + "\n")
    lab = dict(zip(test.pair_id, test.label))
    info = {"H1": {"definition": h1_note, "N": len(h1), "md5": hashlib.md5("\n".join(h1).encode()).hexdigest(), "positives_in_current_test": int(sum(lab.get(p, 0) for p in h1))},
            "H2": {"definition": h2_def, "N": len(h2), "md5": hashlib.md5("\n".join(h2).encode()).hexdigest(), "positives_in_current_test": int(sum(lab.get(p, 0) for p in h2))},
            "overlap_H1_H2": len(set(h1) & set(h2)), "seed": a.seed, "split": a.split}
    json.dump(info, open(out / "eval_sets.json", "w"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
