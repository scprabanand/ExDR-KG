"""Step 14 (Comments 1.6, 1.7): figures redrawn for the revision. Every figure is created at its FINAL printed size (default 7.0 in = full text width),
so font sizes are real: minimum 9 pt everywhere (checked; the script refuses to save a figure that has smaller text), 600 dpi PNG + vector PDF,
colour-blind-safe palette, legends outside the plotting area.

Subcommands
  fig5      Temporal generalisation (old Figure 5): AUROC per year for the compared models, training period (blue) and test period (orange) shaded, dashed
            red line at the cutoff. Legend BELOW the x-axis in one row. Data: either --auroc-csv (year,model,auroc[,lo,hi]) or computed here from the
            per-pair score files of the temporal-split models (--scores "Name=path.csv" ...) plus the temporal test file with a year column (--years-file).
  fig11     Qualitative comparison (old Figure 15, renumbered 11) for ONE case from Step 7 (results/rerun/cases/R-F_cases.json, --case-index): Level 1 (top
            evidence path / TxGNN gene-pathway) and Level 2 explanations from the three systems in three columns, unsupported entities listed under each.
  fig13     Case studies (old Figure 17, renumbered 13): ONE full-width figure per case (fig13a..), evidence path drawn as a chain of boxes, title with drug,
            disease + MONDO id, score, first-evidence year and the literature reference. Cases come from Step 7 (R-F_cases.json); --include-failures adds the
            failure cases, clearly labelled. Captions are written to fig13_captions.txt.
  degree    Supplementary figure: degree distribution by node type on the masked HKG (Step 5 numbers), legend below the axes.
Usage
  python -m exdrkg_rerun.reporting.figures fig5   --root . --scores ExDR-KG=results/rerun/temporal/ExDR-KG.csv TxGNN=results/rerun/temporal/TxGNN.csv R-GCN=results/rerun/temporal/R-GCN.csv --years-file data/splits/temporal_degree_matched/test.csv
  python -m exdrkg_rerun.reporting.figures fig11  --root . --case-index 0 --explanations results/rerun/cases/explanations.jsonl --models ExDR-KG TxGNN+Template BioGPT
  python -m exdrkg_rerun.reporting.figures fig13  --root . --include-failures
  python -m exdrkg_rerun.reporting.figures degree --root . --masked data/processed/masked_standard.pt
Output folder: results/rerun/figures/
"""
import argparse, json, re, sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.stats import rankdata

MIN_PT = 9
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442", "#999999", "#882255", "#44AA99", "#AA4499"]
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, stem, dpi=600):
    small = sorted({(t.get_text()[:30], round(t.get_fontsize(), 1)) for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MIN_PT})
    if small: raise SystemExit(f"{stem}: text below {MIN_PT} pt found: {small[:5]}")
    out.mkdir(parents=True, exist_ok=True); fig.savefig(out / f"{stem}.png", dpi=dpi, bbox_inches="tight"); fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    print(f"saved {out / (stem + '.png')} ({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in at {dpi} dpi)"); plt.close(fig)


def auc(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def boot_auc(y, s, n=1000, seed=42):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); v.append(auc(y[i], s[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fig5(a):
    root = Path(a.root); out = root / a.out
    if a.auroc_csv: D = pd.read_csv(root / a.auroc_csv)
    else:
        Y = pd.read_csv(root / a.years_file); yc = next(c for c in ("first_evidence_year", "year", "first_year") if c in Y.columns)
        Y["pair"] = Y.drug_id.astype(str) + "|" + Y.disease_id.astype(str); Y = Y[["pair", yc]].rename(columns={yc: "year"}); rows = []
        for spec in a.scores:
            name, path = spec.split("=", 1); S = pd.read_csv(root / path); S["pair"] = S.drug_id.astype(str) + "|" + S.disease_id.astype(str); S = S.merge(Y, on="pair", how="left")
            neg_all = S[(S.label == 0)]
            for yr in range(a.first_year, a.last_year + 1):
                pos = S[(S.label == 1) & (S.year == yr)]
                if len(pos) < a.min_pos: continue
                neg = neg_all[neg_all.year == yr] if neg_all.year.notna().any() and (neg_all.year == yr).sum() >= a.min_pos else neg_all      # negatives without a year are used in every year (stated in the caption)
                d = pd.concat([pos, neg]); lo, hi = boot_auc(d.label, d.score); rows.append({"year": yr, "model": name, "auroc": auc(d.label, d.score), "lo": lo, "hi": hi, "n_pos": len(pos)})
        D = pd.DataFrame(rows); D.to_csv(out / "fig5_data.csv", index=False) if out.mkdir(parents=True, exist_ok=True) is None else None
    fig, ax = plt.subplots(figsize=(a.width, 3.6)); ymin, ymax = 0.4, 1.0
    ax.axvspan(D.year.min() - .5, a.cutoff - .5, color="#0072B2", alpha=.08, lw=0); ax.axvspan(a.cutoff - .5, D.year.max() + .5, color="#E69F00", alpha=.12, lw=0); ax.axvline(a.cutoff - .5, color="#C00000", ls="--", lw=1.2)
    for i, (m, g) in enumerate(D.groupby("model", sort=False)):
        g = g.sort_values("year"); ax.plot(g.year, g.auroc, "-o", ms=4, lw=1.6, color=PALETTE[i], label=m)
        if {"lo", "hi"} <= set(g.columns) and g.lo.notna().any(): ax.fill_between(g.year, g.lo, g.hi, color=PALETTE[i], alpha=.15, lw=0)
    ax.text((D.year.min() + a.cutoff) / 2, ymin + .02, "training period", ha="center", fontsize=9, color="#0072B2"); ax.text((a.cutoff + D.year.max()) / 2, ymin + .02, "test period", ha="center", fontsize=9, color="#B06A00")
    ax.set_xlim(D.year.min() - .5, D.year.max() + .5); ax.set_ylim(ymin, ymax); ax.set_xlabel("Year of first evidence"); ax.set_ylabel("AUROC"); ax.grid(alpha=.25)
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, out, "fig05_temporal_generalisation")


def parse_path(txt):
    """'A -[rel]-> B (type) -[rel]-> C' or 'A -> B -> C' -> (nodes, rels)."""
    parts = re.split(r"\s*-\[([^\]]+)\]->\s*", txt.strip())
    if len(parts) >= 3: nodes = [re.sub(r"\s*\([^)]*\)\s*$", "", parts[i]).strip() for i in range(0, len(parts), 2)]; return nodes, parts[1::2]
    nodes = [p.strip() for p in re.split(r"\s*(?:->|→)\s*", txt) if p.strip()]; return nodes, [""] * (len(nodes) - 1)


def chain(ax, nodes, rels, per_row=3, box_w=1.55, gap=1.0, box_h=0.62, row_gap=0.95, colour="#0072B2"):
    def wrap(t): return "\n".join(textwrap.wrap(t, 16)[:3])
    pos = []
    for i, n in enumerate(nodes):
        r, c = divmod(i, per_row); c = c if r % 2 == 0 else per_row - 1 - c; pos.append((c * (box_w + gap), -r * row_gap))
    for i, (x, y) in enumerate(pos):
        fc = colour if i in (0, len(nodes) - 1) else "white"; tc = "white" if i in (0, len(nodes) - 1) else "black"
        ax.add_patch(FancyBboxPatch((x, y - box_h / 2), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=colour, lw=1.3)); ax.text(x + box_w / 2, y, wrap(nodes[i]), ha="center", va="center", fontsize=9, color=tc)
    for i in range(len(nodes) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            s, e = (x0 + box_w, x1) if x1 > x0 else (x0, x1 + box_w); ax.annotate("", xy=(e, y1), xytext=(s, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color="#444")); ax.text((s + e) / 2, y0 + 0.08, rels[i], ha="center", va="bottom", fontsize=9, style="italic")
        else:
            ax.annotate("", xy=(x1 + box_w / 2, y1 + box_h / 2), xytext=(x0 + box_w / 2, y0 - box_h / 2), arrowprops=dict(arrowstyle="->", lw=1.3, color="#444")); ax.text(x0 + box_w / 2 + 0.08, (y0 + y1) / 2, rels[i], ha="left", va="center", fontsize=9, style="italic")
    xs = [p[0] for p in pos]; ys = [p[1] for p in pos]; ax.set_xlim(min(xs) - .15, max(xs) + box_w + .15); ax.set_ylim(min(ys) - box_h / 2 - .1, max(ys) + box_h / 2 + .35); ax.axis("off")


def cases_of(a):
    C = json.load(open(Path(a.root) / a.cases))
    return C


def fig13(a):
    root = Path(a.root); out = root / a.out; C = cases_of(a); cap = []
    sel = [c for c in C if c["case_type"] == "success"] + ([c for c in C if c["case_type"] == "failure"] if a.include_failures else [])
    for i, c in enumerate(sel):
        nodes, rels = parse_path(c["top_evidence_path"]); rows = (len(nodes) + 2) // 3; fig, ax = plt.subplots(figsize=(a.width, 0.95 * rows + 1.5))
        tag = "Failure case: " if c["case_type"] == "failure" else ""
        ax.set_title(f"({chr(97 + i)}) {tag}{c.get('drug_name') or c['drug_id']} for {c.get('disease_name') or c['disease_id']} ({c['mondo_id']}); score {c['score']:.2f}; first evidence {c['first_evidence_year']}", fontsize=10, loc="left")
        chain(ax, nodes, rels, colour="#C00000" if c["case_type"] == "failure" else "#0072B2")
        ref = f"{c.get('reference_type', '')} {c.get('reference_id', '')} ({c.get('reference_year', '')})".strip(); fig.text(0.01, 0.01, f"Reference: {ref}", fontsize=9, ha="left", va="bottom")
        save(fig, out, f"fig13{chr(97 + i)}_case_{i + 1}")
        cap.append(f"Figure 13{chr(97 + i)}. {tag}{c.get('drug_name') or c['drug_id']} - {c.get('disease_name') or c['disease_id']} ({c['mondo_id']}). Predicted score {c['score']:.3f} (rank {c['rank_among_new_positives']} of {c['n_new_positives']} "
                   f"post-{a.min_year - 1} positives). Absent from the training HKG: {c['absent_from_training_HKG']}. Evidence: {c['reference_type']} {c['reference_id']} ({c['reference_year']}). " + (f"Reason ranked low: {c['failure_reason']}" if c["case_type"] == "failure" else ""))
    (out / "fig13_captions.txt").write_text("\n\n".join(cap) + "\n")


def fig11(a):
    root = Path(a.root); out = root / a.out; C = json.load(open(root / a.cases)); c = C[a.case_index]; pid = f"{c['drug_id']}|{c['disease_id']}"; models = a.models
    E = {}
    for j in map(json.loads, open(root / a.explanations)):
        if j["pair_id"] == pid and j["model"] in models: E[j["model"]] = j["explanation"]
    uns = {}
    if a.per_explanation and (root / a.per_explanation).exists():
        P = pd.read_csv(root / a.per_explanation); P = P[P.pair_id == pid]; uns = {r.model: (str(r.unsupported) if isinstance(r.unsupported, str) else "") for r in P.itertuples()}
    tx = {}
    if a.txgnn_paths and (root / a.txgnn_paths).exists():
        tx = {j["pair_id"]: j for j in map(json.loads, open(root / a.txgnn_paths))}.get(pid, {})
    L1 = {models[0]: c["top_evidence_path"], **({models[1]: f"{tx['gene']} -> {tx['pathway']}"} if tx and len(models) > 1 else {}), **({models[2]: "no pathway-level output"} if len(models) > 2 else {})}
    col_in = a.width / len(models); wrap_w = int((col_in - 0.2) / 0.078); rows = [("Level 1: pathway", L1), ("Level 2: mechanism", E)]
    heights = [max(len(textwrap.wrap(str(d.get(m, "n/a")), wrap_w)) for m in models) * 0.155 + 0.5 for _, d in rows]; heights[1] += 0.55
    fig, axes = plt.subplots(2, len(models), figsize=(a.width, sum(heights) + 0.6), gridspec_kw={"height_ratios": heights})
    for r, (lab, d) in enumerate(rows):
        for k, m in enumerate(models):
            ax = axes[r][k]; ax.axis("off"); ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#F7F7F7", ec="#999", lw=1))
            if r == 0: ax.set_title(m, fontsize=10, fontweight="bold")
            ax.text(0.03, 0.97, "\n".join(textwrap.wrap(str(d.get(m, "n/a")), wrap_w)), transform=ax.transAxes, va="top", ha="left", fontsize=9)
            if r == 1 and m in uns: ax.text(0.03, 0.02, ("unsupported entities: " + (uns[m].replace(";", ", ") or "none"))[:wrap_w * 2], transform=ax.transAxes, va="bottom", fontsize=9, color="#C00000")
        axes[r][0].text(-0.02, 0.5, lab, transform=axes[r][0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"{c.get('drug_name') or c['drug_id']} - {c.get('disease_name') or c['disease_id']} ({c['mondo_id']}), score {c['score']:.2f}", fontsize=10, y=1.0); fig.tight_layout(); save(fig, out, "fig11_qualitative_comparison")


def degree(a):
    import torch
    from exdrkg_rerun.graph_analysis.hkg_statistics import load_graph
    root = Path(a.root); out = root / a.out; kg = torch.load(root / a.masked, weights_only=False, map_location="cpu"); g, types, names, _, _ = load_graph(kg); deg = np.array(g.degree())
    fig, ax = plt.subplots(1, 2, figsize=(a.width, 3.9))
    for i, t in enumerate(names):
        x = deg[(types == t) & (deg > 0)]
        if len(x) == 0: continue
        bins = np.unique(np.logspace(0, np.log10(x.max() + 1), 25).astype(int)); bins = np.append(bins, x.max() + 1); h, e = np.histogram(x, bins=bins, density=True); c = np.sqrt(e[:-1] * e[1:]); m = h > 0
        ax[0].loglog(c[m], h[m], "-o", ms=3, lw=1.3, color=PALETTE[i % 12], label=t); s = np.sort(x); ax[1].loglog(s, 1 - np.arange(len(s)) / len(s), lw=1.5, color=PALETTE[i % 12])
    ax[0].set_xlabel("degree k"); ax[0].set_ylabel("P(k), log-binned"); ax[1].set_xlabel("degree k"); ax[1].set_ylabel("P(K >= k)")
    for x_ in ax: x_.grid(alpha=.25, which="both")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02)); fig.tight_layout(rect=(0, 0.2, 1, 1)); save(fig, out, "figS1_degree_distribution")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("fig5", "fig11", "fig13", "degree"):
        p = sub.add_parser(n); p.add_argument("--root", default="."); p.add_argument("--out", default="results/rerun/figures"); p.add_argument("--width", type=float, default=7.0, help="final text width in inches")
    f = sub.choices["fig5"]; f.add_argument("--auroc-csv", default=None); f.add_argument("--scores", nargs="*", default=[]); f.add_argument("--years-file", default=None); f.add_argument("--cutoff", type=int, default=2020)
    f.add_argument("--first-year", type=int, default=2010); f.add_argument("--last-year", type=int, default=2024); f.add_argument("--min-pos", type=int, default=5)
    for n in ("fig11", "fig13"): sub.choices[n].add_argument("--cases", default="results/rerun/cases/R-F_cases.json")
    g = sub.choices["fig11"]; g.add_argument("--case-index", type=int, default=0); g.add_argument("--explanations", required=True); g.add_argument("--models", nargs=3, required=True)
    g.add_argument("--per-explanation", default="results/rerun/cases/per_explanation.csv"); g.add_argument("--txgnn-paths", default="data/txgnn_top_paths.jsonl")
    h = sub.choices["fig13"]; h.add_argument("--include-failures", action="store_true"); h.add_argument("--min-year", type=int, default=2020)
    sub.choices["degree"].add_argument("--masked", required=True)
    a = ap.parse_args(); {"fig5": fig5, "fig11": fig11, "fig13": fig13, "degree": degree}[a.cmd](a)


if __name__ == "__main__":
    main()
