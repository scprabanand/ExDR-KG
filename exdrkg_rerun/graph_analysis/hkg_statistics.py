"""Step 5: HKG statistics for Reviewer 1 (Comment 1.5): connectivity, shortest paths, degree by node type, degree plot.

Everything is computed on the MASKED training HKG (validation and test indication edges, reverses and duplicates removed;
build it with masked_kg.py) and edges are treated as undirected. The graph is loaded into python-igraph.

Tables produced
  R-E1  global connectivity: number of components, largest component (nodes, % of all nodes), next largest sizes, isolated
        nodes (also per node type), average / median / maximum shortest-path length ESTIMATED from 10,000 random node pairs
        inside the largest component (random.seed(42), exactly the sampling of the rerun plan), optional exact diameter.
  R-E2  degree by node type for all 12 types: count, mean, median, maximum (plus p90 and p99). Two definitions are written:
        `undirected_simple` (each neighbour counted once; the one used with the connectivity numbers) and `directed_in_out`
        (in + out edge count, the convention of the build log).
Figure
  results/rerun/stats/fig_degree_distribution.{png,pdf}: log-log degree distribution, one line per node type. Left panel:
  log-binned probability P(k); right panel: complementary cumulative distribution P(K >= k). 600 dpi PNG for the journal.

The text to put in the manuscript is written to results/rerun/stats/methods_sentence.txt and states that shortest-path values
are estimated from 10,000 sampled pairs.

Usage
  python -m exdrkg_rerun.graph_analysis.hkg_statistics --root . --masked data/processed/masked_standard.pt [--exact-diameter]
Needs: python-igraph, numpy, pandas, matplotlib, torch, torch_geometric.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import igraph as ig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

# colour-blind-safe qualitative palette (Okabe-Ito extended); one colour per node type, fixed order
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442", "#999999", "#882255", "#44AA99", "#AA4499"]


def load_graph(kg):
    """Return igraph.Graph (undirected, simple), node-type array, type names, directed in+out degree array."""
    off, o = {}, 0
    for t in kg.node_types:
        off[t] = o; o += kg[t].num_nodes
    n = o; src, dst = [], []
    for et in kg.edge_types:
        s, _, d = et; ei = kg[et].edge_index.cpu().numpy()
        src.append(ei[0] + off[s]); dst.append(ei[1] + off[d])
    src, dst = np.concatenate(src), np.concatenate(dst)
    directed_edges = len(src)
    dirdeg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    g = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.simplify(multiple=True, loops=True)
    types = np.concatenate([[t] * kg[t].num_nodes for t in kg.node_types])
    return g, types, list(kg.node_types), dirdeg, directed_edges


def connectivity(g, types, sample_pairs=10000, seed=42, exact_diameter=False):
    cc = g.connected_components(); sizes = sorted(cc.sizes(), reverse=True); deg = np.array(g.degree())
    gc = cc.giant(); random.seed(seed); n = gc.vcount()
    pairs = [random.sample(range(n), 2) for _ in range(sample_pairs)]          # exactly the sampling of the plan
    d = np.array([gc.distances(s, t)[0][0] for s, t in pairs], dtype=float)
    iso = {t: int(((deg == 0) & (types == t)).sum()) for t in np.unique(types)}
    res = {"nodes": g.vcount(), "undirected_simple_edges": g.ecount(), "components": len(sizes), "largest_component_nodes": int(sizes[0]),
           "largest_component_pct": 100 * sizes[0] / g.vcount(), "next_largest_sizes": [int(x) for x in sizes[1:6]],
           "components_of_size_1": int(sum(1 for x in sizes if x == 1)), "isolated_nodes": int((deg == 0).sum()), "isolated_by_type": iso,
           "asp_mean": float(d.mean()), "asp_median": float(np.median(d)), "asp_max_sampled": float(d.max()), "asp_sampled_pairs": sample_pairs,
           "asp_distance_histogram": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}}
    if exact_diameter:
        t0 = time.time(); res["diameter_exact_largest_component"] = int(gc.diameter(directed=False)); res["diameter_seconds"] = time.time() - t0
    else:  # cheap double-sweep lower bound so a diameter estimate always exists
        v = random.randrange(n); far = int(np.argmax(gc.distances(v)[0])); res["diameter_double_sweep_lower_bound"] = int(max(gc.distances(far)[0]))
    return res


def degree_table(deg, types, names):
    rows = []
    for t in names:
        x = deg[types == t]
        rows.append({"node_type": t, "count": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "max": int(x.max()),
                     "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)), "zero_degree": int((x == 0).sum())})
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def plot_degrees(deg, types, names, out_png, out_pdf):
    order = list(names); col = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(order)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for t in order:
        x = deg[(types == t) & (deg > 0)]
        if len(x) == 0: continue
        bins = np.unique(np.logspace(0, np.log10(x.max() + 1), 25).astype(int)); bins = np.append(bins, x.max() + 1)
        h, e = np.histogram(x, bins=bins, density=True); c = np.sqrt(e[:-1] * e[1:]); m = h > 0
        ax[0].loglog(c[m], h[m], "-o", ms=3, lw=1.3, color=col[t], label=f"{t} (n={len(x):,})")
        s = np.sort(x); ccdf = 1 - np.arange(len(s)) / len(s); ax[1].loglog(s, ccdf, lw=1.5, color=col[t])
    ax[0].set_xlabel("degree k (log scale)"); ax[0].set_ylabel("P(k), log-binned"); ax[0].set_title("Degree distribution")
    ax[1].set_xlabel("degree k (log scale)"); ax[1].set_ylabel("P(K >= k)"); ax[1].set_title("Complementary cumulative distribution")
    for a_ in ax: a_.grid(alpha=0.25, which="both"); a_.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left", ncol=1)
    fig.tight_layout(); fig.savefig(out_png, dpi=600); fig.savefig(out_pdf); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--masked", required=True); ap.add_argument("--out", default="results/rerun/stats")
    ap.add_argument("--pairs", type=int, default=10000); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--exact-diameter", action="store_true")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    kg = torch.load(root / a.masked, weights_only=False, map_location="cpu")
    g, types, names, dirdeg, n_dir = load_graph(kg); print(f"loaded: {g.vcount():,} nodes, {n_dir:,} directed edges, {g.ecount():,} undirected simple edges", flush=True)
    E1 = connectivity(g, types, a.pairs, a.seed, a.exact_diameter); E1["directed_edges_in_file"] = n_dir
    und = np.array(g.degree()); E2u = degree_table(und, types, names); E2d = degree_table(dirdeg, types, names)
    json.dump({"R-E1": E1, "masked_graph": a.masked, "seed": a.seed, "seconds": time.time() - t0}, open(out / "R-E1.json", "w"), indent=1)
    E2u.to_csv(out / "R-E2_undirected_simple.csv", index=False); E2d.to_csv(out / "R-E2_directed_in_out.csv", index=False)
    (out / "R-E2_undirected_simple.md").write_text(E2u.round(2).to_markdown(index=False)); (out / "R-E2_directed_in_out.md").write_text(E2d.round(2).to_markdown(index=False))
    plot_degrees(und, types, names, out / "fig_degree_distribution.png", out / "fig_degree_distribution.pdf")
    dia = E1.get("diameter_exact_largest_component", None)
    sent = (f"Treating edges as undirected, the masked HKG has {E1['nodes']:,} nodes and {E1['undirected_simple_edges']:,} unique undirected edges in "
            f"{E1['components']:,} connected components. The largest component contains {E1['largest_component_nodes']:,} nodes "
            f"({E1['largest_component_pct']:.2f}%); the next largest components have {', '.join(str(x) for x in E1['next_largest_sizes'][:3])} nodes and "
            f"{E1['isolated_nodes']:,} nodes are isolated. Shortest-path lengths were estimated from {E1['asp_sampled_pairs']:,} random node pairs in the largest "
            f"component (seed {a.seed}): mean {E1['asp_mean']:.2f}, median {E1['asp_median']:.0f}, maximum sampled distance {E1['asp_max_sampled']:.0f}"
            + (f"; the exact diameter of the largest component is {dia}." if dia is not None else " (the maximum is a lower bound on the diameter)."))
    (out / "methods_sentence.txt").write_text(sent + "\n"); print(json.dumps(E1, indent=1)); print(E2u.round(2).to_string(index=False)); print("\n" + sent)


if __name__ == "__main__":
    main()
