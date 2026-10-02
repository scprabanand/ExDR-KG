"""Rebuild the negative pool so that it is leakage-free, then write a new split directory.

Problems in the shipped splits (see R-K_step1_results.md)
  * 4,674 of 11,104 unique standard-test negative pairs (42%) also appear as training negatives;
  * 22,034 / 416 / 396 repeated (drug, disease) rows in train / val / test;
  * negatives are uniform random corruptions (the manuscript says type-constrained).

What this script guarantees (asserted before writing)
  * positives are copied unchanged (same pairs, same split), so results stay comparable;
  * exactly `ratio` negatives per positive in every split (default 100, as in the manuscript);
  * every negative pair is unique within its split and appears in NO other split;
  * no negative is a known indication anywhere in the knowledge graph (train, val or test positive);
  * for zero-shot style splits, negatives only use the drugs/diseases that the original split file uses for that part,
    so held-out diseases never leak into training negatives.

Negative strategies
  random          uniform head/tail corruption (like the original, but disjoint and unique)
  degree_matched  the corrupted entity is drawn among entities with a similar number of known indications (log-degree
                  bucket of the positive's drug/disease), which removes the "popular drugs are always positive"
                  shortcut; recommended for the reported evaluation
  Use both: report `random` as the historical setting and `degree_matched` as the harder, fairer one.

Usage
  python -m exdrkg_rerun.data.rebuild_splits --root . --split standard --strategy degree_matched --seed 42
  -> data/splits/standard_degree_matched/{train,val,test}.csv + split_stats.json + neg_pool.md5
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ORDER = ["test", "val", "train"]      # test is filled first so it gets the cleanest pool


def bucket(counts: dict, keys, n_buckets=8):
    """Assign each entity to a log-count bucket (quantile based)."""
    v = np.array([counts.get(k, 0) for k in keys], dtype=float)
    q = np.unique(np.quantile(np.log1p(v), np.linspace(0, 1, n_buckets + 1)[1:-1]))
    b = np.searchsorted(q, np.log1p(v), side="right")
    return dict(zip(keys, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard")
    ap.add_argument("--strategy", choices=["random", "degree_matched"], default="degree_matched")
    ap.add_argument("--ratio", type=int, default=100); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); root = Path(a.root); rng = np.random.default_rng(a.seed)
    src = root / "data/splits" / a.split
    parts = {p: pd.read_csv(src / f"{p}.csv") for p in ORDER if (src / f"{p}.csv").exists()}
    pos = {p: df[df.label == 1][["drug_id", "disease_id", "relation"]].drop_duplicates(["drug_id", "disease_id"]) for p, df in parts.items()}
    all_pos = set()
    for p in pos.values(): all_pos |= set(zip(p.drug_id, p.disease_id))
    n_drug_pos, n_dis_pos = {}, {}
    for d, s in all_pos: n_drug_pos[d] = n_drug_pos.get(d, 0) + 1; n_dis_pos[s] = n_dis_pos.get(s, 0) + 1
    # entity universes per split, taken from the ORIGINAL split file (protects zero-shot held-out diseases)
    drugs_in = {p: sorted(set(df.drug_id)) for p, df in parts.items()}; dis_in = {p: sorted(set(df.disease_id)) for p, df in parts.items()}
    used = set(all_pos); neg_rows = {}
    for p in ORDER:
        if p not in parts: continue
        drugs, diseases = drugs_in[p], dis_in[p]
        db, sb = bucket(n_drug_pos, drugs), bucket(n_dis_pos, diseases)
        by_db, by_sb = {}, {}
        for d in drugs: by_db.setdefault(db[d], []).append(d)
        for s in diseases: by_sb.setdefault(sb[s], []).append(s)
        need = a.ratio * len(pos[p]); rows = []; tries = 0
        P = list(zip(pos[p].drug_id, pos[p].disease_id))
        while len(rows) < need:
            d, s = P[rng.integers(len(P))]; tries += 1
            if tries > 500 * need: raise RuntimeError(f"cannot find {need} unique negatives for {p}; lower --ratio")
            head = rng.random() < 0.5
            if a.strategy == "random":
                cand = (drugs[rng.integers(len(drugs))], s) if head else (d, diseases[rng.integers(len(diseases))])
            else:
                if head:
                    pool = by_db.get(db.get(d, next(iter(by_db))), drugs); cand = (pool[rng.integers(len(pool))], s)
                else:
                    pool = by_sb.get(sb.get(s, next(iter(by_sb))), diseases); cand = (d, pool[rng.integers(len(pool))])
            if cand in used: continue
            used.add(cand); rows.append((cand[0], cand[1], "corrupt_head" if head else "corrupt_tail"))
        neg_rows[p] = pd.DataFrame(rows, columns=["drug_id", "disease_id", "neg_type"])
    out = Path(a.out or root / "data/splits" / f"{a.split}_{a.strategy}"); out.mkdir(parents=True, exist_ok=True)
    frames, stats = {}, {}
    for p in parts:
        pp = pos[p].assign(label=1, neg_type="positive", split=p)
        nn = neg_rows[p].assign(relation="none", label=0, split=p)
        df = pd.concat([pp, nn], ignore_index=True)[["drug_id", "disease_id", "relation", "label", "neg_type", "split"]]
        df = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True); frames[p] = df
        stats[p] = {"total": len(df), "positives": int(df.label.sum()), "negatives": int((df.label == 0).sum())}
    # ---- guarantees
    keys = {p: set(zip(f.drug_id, f.disease_id)) for p, f in frames.items()}
    for p, f in frames.items():
        assert not f.duplicated(["drug_id", "disease_id"]).any(), f"duplicate pairs in {p}"
        assert stats[p]["negatives"] == a.ratio * stats[p]["positives"], f"ratio broken in {p}"
    ps = list(frames)
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)): assert not (keys[ps[i]] & keys[ps[j]]), f"pair overlap {ps[i]}/{ps[j]}"
    for p, f in frames.items():
        neg = set(zip(f[f.label == 0].drug_id, f[f.label == 0].disease_id)); assert not (neg & all_pos), f"negative equals a known indication in {p}"
        f.to_csv(out / f"{p}.csv", index=False)
    negpool = pd.concat([f[f.label == 0].assign(part=p) for p, f in frames.items()]).sort_values(["part", "drug_id", "disease_id"])
    h = hashlib.md5(negpool[["part", "drug_id", "disease_id"]].to_csv(index=False).encode()).hexdigest()
    (out / "neg_pool.md5").write_text(h + "\n")
    json.dump({"source_split": a.split, "strategy": a.strategy, "ratio": a.ratio, "seed": a.seed, "stats": stats, "neg_pool_md5": h,
               "guarantees": ["unique pairs per split", "no pair shared across splits", "no negative is a known indication", "positives unchanged"]},
              open(out / "split_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1), "\nneg pool md5", h, "\nwritten to", out)


if __name__ == "__main__":
    main()
