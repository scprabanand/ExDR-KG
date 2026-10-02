"""Step 6: path availability for evaluation pairs on the MASKED HKG (Table R-E4, Comment 1.5).

Purpose: show how often the retriever can find evidence, and check that path existence alone does not reveal the label.
Positives and negatives are ALWAYS reported separately.

Same search as the retriever
  The retriever (src/models/evidence_retriever.py, EvidencePathRetriever._enumerate_paths) enumerates ALL simple DIRECTED paths
  from the drug to the disease with at most `max_hops` = 4 edges, following edge_index direction (reverse relations are
  ordinary edges of their own type), with no meta-path filter, and it keeps parallel edges (two relations between the same two
  nodes give two candidate paths with the same node sequence). This script reproduces exactly that search, but much faster
  (numba DFS on a CSR graph, pruned with reverse-BFS distances to the disease, which never removes a valid path) so it can be
  run for every test pair. For each pair it records
    * has_path       at least one path within 4 hops                                       (retriever finds evidence)
    * n_paths        number of candidate paths BEFORE MMR (retriever semantics, parallel edges counted); capped at --cap
    * n_paths_unique the same on the graph with parallel edges merged (distinct node sequences)
    * n_len1..n_len4 candidate paths per length
    * shortest_dir   directed shortest-path length (hops, up to 6; -1 if unreachable)
    * capped         True if the cap was hit (the count is then a lower bound)
  Path counts use the retriever's own semantics; the retriever itself would need minutes per hub-heavy pair because it rebuilds
  its adjacency in Python on every call, which is why the counts are computed here instead of by calling it.

Jobs (--jobs, default below): "<split>:<part>" pairs, each evaluated on the graph masked for THAT split
  standard:test       standard test positives and negatives
  zero_shot:test      zero-shot test positives and negatives
  temporal:test       temporal-split positives (needs data/splits/temporal/test.csv from a rebuilt temporal split; skipped if absent)
The temporal graph should also be restricted to edges dated before the cutoff (masked_kg.py does that only if an edge-timestamp
table is supplied); without it the temporal row is only a masked-graph result and must be labelled that way.

Leakage check (does path existence reveal the label?)
  For every job the script writes AUROC of three label-free scores: has_path, -shortest_dir, log1p(n_paths), with 95% CIs, and a
  Fisher exact test for has_path x label. Values near 0.5 support "path existence alone does not reveal the label"; values far
  above 0.5 mean the benchmark is solvable without a model and must be reported as a limitation.

Usage
  python -m exdrkg_rerun.graph_analysis.path_availability --root . --jobs standard:test zero_shot:test temporal:test --cap 1000000
Outputs (results/rerun/paths/): pairs_<split>_<part>.csv, R-E4.csv, R-E4.md, R-E4_detail.json, leakage_check.csv
Needs: numba, numpy, pandas, scipy, scikit-learn, torch, torch_geometric.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from numba import njit, prange
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from exdrkg_rerun.data.masked_kg import build_masked_kg

MAX_HOPS = 4
BFS_DEPTH = 6
UNREACH = 127


def to_csr(kg, dedupe):
    """Directed CSR over global ids (node types in SORTED order, as in the retriever). Parallel edges kept unless dedupe."""
    types = sorted(kg.node_types); off, o = {}, 0
    for t in types: off[t] = o; o += kg[t].num_nodes
    n = o; src, dst = [], []
    for et in kg.edge_types:
        s, _, d = et; ei = kg[et].edge_index.cpu().numpy(); src.append(ei[0] + off[s]); dst.append(ei[1] + off[d])
    src, dst = np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64)
    if dedupe:
        key = np.unique(src * n + dst); src, dst = key // n, key % n
    def csr(a, b):
        order = np.argsort(a, kind="stable"); a, b = a[order], b[order]
        indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, a + 1, 1); return np.cumsum(indptr), b
    ip, ix = csr(src, dst); rip, rix = csr(dst, src)
    return n, off, ip, ix, rip, rix


@njit(cache=True)
def bfs_reverse(rip, rix, dst, n, maxd):
    d = np.full(n, 127, np.int8); d[dst] = 0
    frontier = np.empty(n, np.int64); nxt = np.empty(n, np.int64); frontier[0] = dst; fl = 1
    for depth in range(1, maxd + 1):
        nl = 0
        for i in range(fl):
            u = frontier[i]
            for k in range(rip[u], rip[u + 1]):
                v = rix[k]
                if d[v] == 127:
                    d[v] = depth; nxt[nl] = v; nl += 1
        if nl == 0: break
        for i in range(nl): frontier[i] = nxt[i]
        fl = nl
    return d


@njit(parallel=True, cache=True)
def all_rd(rip, rix, dis_nodes, n, maxd):
    out = np.empty((len(dis_nodes), n), np.int8)
    for i in prange(len(dis_nodes)): out[i] = bfs_reverse(rip, rix, dis_nodes[i], n, maxd)
    return out


@njit(cache=True)
def dfs_count(ip, ix, src, dst, rd, max_hops, cap, counts):
    """Count simple directed paths src -> dst with <= max_hops edges (retriever semantics). counts[l] = paths with l edges."""
    path = np.empty(max_hops + 1, np.int64); ptr = np.empty(max_hops + 1, np.int64)
    depth = 0; path[0] = src; ptr[0] = ip[src]; total = 0
    while depth >= 0:
        node = path[depth]
        if ptr[depth] >= ip[node + 1]:
            depth -= 1; continue
        nbr = ix[ptr[depth]]; ptr[depth] += 1
        seen = False
        for i in range(depth + 1):
            if path[i] == nbr: seen = True; break
        if seen: continue
        if nbr == dst:
            counts[depth + 1] += 1; total += 1
            if total >= cap: return total, True
            continue
        remaining = max_hops - (depth + 1)
        if remaining <= 0 or rd[nbr] > remaining: continue
        depth += 1; path[depth] = nbr; ptr[depth] = ip[nbr]
    return total, False


@njit(parallel=True, cache=True)
def count_pairs(ip, ix, srcs, dis_idx, dis_nodes, rd, max_hops, cap):
    m = len(srcs); tot = np.zeros(m, np.int64); capped = np.zeros(m, np.bool_); lens = np.zeros((m, max_hops + 1), np.int64)
    for i in prange(m):
        c = np.zeros(max_hops + 1, np.int64)
        t, cp = dfs_count(ip, ix, srcs[i], dis_nodes[dis_idx[i]], rd[dis_idx[i]], max_hops, cap, c)
        tot[i] = t; capped[i] = cp; lens[i] = c
    return tot, capped, lens


def boot_auc(y, s, n=1000, seed=42):
    rng = np.random.default_rng(seed); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; v = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); v.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run_job(root, split, part, cap, out):
    f = root / "data/splits" / split / f"{part}.csv"
    if not f.exists(): print(f"SKIP {split}:{part} (no {f})"); return None
    df = pd.read_csv(f); kg, info = build_masked_kg(root, split); t0 = time.time()
    n, off, ip, ix, rip, rix = to_csr(kg, dedupe=False); _, _, ipu, ixu, _, _ = to_csr(kg, dedupe=True)
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}; sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    dis_nodes = (np.arange(len(sid)) + off["disease"]).astype(np.int64)
    rd = all_rd(rip, rix, dis_nodes, n, BFS_DEPTH)
    srcs = (df.drug_id.map(did).values + off["drug"]).astype(np.int64); dis_idx = df.disease_id.map(sid).values.astype(np.int64)
    tot, capped, lens = count_pairs(ip, ix, srcs, dis_idx, dis_nodes, rd, MAX_HOPS, cap)
    totu, _, _ = count_pairs(ipu, ixu, srcs, dis_idx, dis_nodes, rd, MAX_HOPS, cap)
    short = rd[dis_idx, srcs].astype(int); short = np.where(short == UNREACH, -1, short)
    res = df.assign(has_path=(tot > 0).astype(int), n_paths=tot, n_paths_unique=totu, capped=capped, shortest_dir=short,
                    **{f"n_len{l}": lens[:, l] for l in range(1, MAX_HOPS + 1)})
    res.to_csv(out / f"pairs_{split}_{part}.csv", index=False); print(f"{split}:{part}: {len(df)} pairs in {time.time() - t0:.0f}s (masked: {info})", flush=True)
    return res, info


def summarise(res, split, part, info):
    rows, det = [], {}
    for lab, name in [(1, "positives"), (0, "negatives")]:
        g = res[res.label == lab]
        if g.empty: continue
        k = int(g.has_path.sum()); with_p = g[g.has_path == 1]
        sp = g[g.shortest_dir > 0]
        rows.append({"Pair group": f"{split} {part} {name}", "Pairs (N)": len(g), "With >= 1 path, k (%)": f"{k} ({100 * k / len(g):.1f}%)",
                     "Median candidate paths (all pairs)": float(g.n_paths.median()), "Median candidate paths (pairs with a path)": float(with_p.n_paths.median()) if k else None,
                     "Mean shortest path length": round(float(sp.shortest_dir.mean()), 2) if len(sp) else None,
                     "Capped pairs": int(g.capped.sum())})
        det[f"{split}_{part}_{name}"] = {"N": len(g), "k_with_path": k, "pct": 100 * k / len(g), "shortest_hist": {int(a): int(b) for a, b in zip(*np.unique(g.shortest_dir, return_counts=True))},
                                         "n_paths_quantiles": {q: float(np.percentile(g.n_paths, q)) for q in (5, 25, 50, 75, 95)}, "unreachable": int((g.shortest_dir < 0).sum()),
                                         "n_paths_unique_median": float(g.n_paths_unique.median())}
        if "neg_type" in g and lab == 0:
            det[f"{split}_{part}_{name}"]["by_neg_type"] = {str(t): {"N": int(len(x)), "pct_with_path": float(100 * x.has_path.mean())} for t, x in g.groupby("neg_type")}
    return rows, det


def leakage(res, split, part):
    y = res.label.values; out = []
    if len(np.unique(y)) < 2: return out
    scores = {"has_path": res.has_path.values.astype(float), "-shortest_dir (unreachable = worst)": -np.where(res.shortest_dir.values < 0, 99, res.shortest_dir.values).astype(float),
              "log1p(n_paths)": np.log1p(res.n_paths.values.astype(float)), "log1p(n_paths) among pairs with a path": np.where(res.has_path.values == 1, np.log1p(res.n_paths.values.astype(float)), -1.0)}
    tab = [[int(((res.has_path == 1) & (res.label == 1)).sum()), int(((res.has_path == 0) & (res.label == 1)).sum())],
           [int(((res.has_path == 1) & (res.label == 0)).sum()), int(((res.has_path == 0) & (res.label == 0)).sum())]]
    _, p = fisher_exact(tab)
    for k, s in scores.items():
        lo, hi = boot_auc(y, s); out.append({"job": f"{split}:{part}", "label-free score": k, "AUROC": round(roc_auc_score(y, s), 3), "CI95": f"{lo:.3f}-{hi:.3f}", "fisher_p_has_path_x_label": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--jobs", nargs="*", default=["standard:test", "zero_shot:test", "temporal:test"])
    ap.add_argument("--cap", type=int, default=1_000_000); ap.add_argument("--out", default="results/rerun/paths")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    rows, det, leak = [], {}, []
    for j in a.jobs:
        split, part = j.split(":"); r = run_job(root, split, part, a.cap, out)
        if r is None: continue
        res, info = r; rr, dd = summarise(res, split, part, info); rows += rr; det.update(dd); leak += leakage(res, split, part)
    T = pd.DataFrame(rows); T.to_csv(out / "R-E4.csv", index=False); (out / "R-E4.md").write_text(T.to_markdown(index=False))
    L = pd.DataFrame(leak); L.to_csv(out / "leakage_check.csv", index=False)
    json.dump({"max_hops": MAX_HOPS, "cap": a.cap, "semantics": "simple directed paths, parallel edges counted, no meta-path filter (retriever _enumerate_paths)", "groups": det},
              open(out / "R-E4_detail.json", "w"), indent=1)
    print(T.to_string(index=False)); print(); print(L.to_string(index=False) if len(L) else "no leakage rows")


if __name__ == "__main__":
    main()
