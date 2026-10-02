"""Step 9: assemble Table R-C (depth ablation) and the per-layer MAD table from results/rerun/<split>/depth_L*.json.

Checks and rules
  * L = 6 metrics are taken from the V1 scores file (results/rerun/main/scores/V1.csv, written by Step 3), so they equal V1 in Table R-B
    by construction; the head re-run done with depth_ablation.py --layers 6 is reported next to it as `L6_head_rerun_delta` (a large gap
    means the V1 head recipe here differs from the one used for V1 and must be reconciled before submission).
  * Every depth must share the same MAD sample (md5 identical) and the same number of nodes; otherwise the script stops.
  * Reads "MAD trend": relative change from the shallowest to the deepest model and from L=6 to L=10, printed as plain numbers only
    (no claim of over-smoothing is written; a sharp fall is the plan's criterion, the reader decides).
Usage: python -m exdrkg_rerun.evaluation.depth_summary --root . --split standard_degree_matched [--v1-scores results/rerun/main/scores/V1.csv]
Outputs: results/rerun/<split>/R-C.{csv,md,tex}, R-C_per_layer_mad.csv, R-C_notes.txt, fig_mad_vs_depth.png (600 dpi)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard_degree_matched")
    ap.add_argument("--v1-scores", default="results/rerun/main/scores/V1.csv"); a = ap.parse_args()
    root = Path(a.root); d = root / "results" / "rerun" / a.split; R = {}
    for f in sorted(d.glob("depth_L*.json")): j = json.load(open(f)); R[j["layers"]] = j
    if not R: raise SystemExit("no depth_L*.json found")
    md5s = {j["mad_sample_md5"] for j in R.values()}; nn_ = {j["n_nodes_total"] for j in R.values()}
    if len(md5s) > 1 or len(nn_) > 1: raise SystemExit(f"MAD samples differ across depths: md5 {md5s}, n_nodes {nn_}")
    notes = []; rows = []
    v1 = Path(root / a.v1_scores); v1_metrics = v1_ci = None
    if v1.exists():
        s = pd.read_csv(v1); v1_metrics = compute_metrics(s, s.score.values); v1_ci = bootstrap_ci(s, s.score.values)
    else: notes.append(f"{a.v1_scores} not found: L=6 row uses the head re-run from depth_ablation.py, which is NOT guaranteed to equal V1")
    for L in sorted(R):
        j = R[L]; m, ci = j["metrics"], j["ci95"]; src = "this run"
        if L == 6 and v1_metrics is not None:
            delta = {k: round(float(j["metrics"][k]) - float(v1_metrics[k]), 4) for k in ("AUROC", "AUPRC") if k in v1_metrics}
            notes.append(f"L=6 head re-run minus V1: {delta}"); m, ci, src = v1_metrics, v1_ci, "V1 scores (existing checkpoint)"
        rows.append({"Layers": L, "AUROC": m["AUROC"], "AUPRC": m["AUPRC"], "Hit@10": m.get("Hit@10"), "MRR": m.get("MRR"), "MAD": j["MAD"], "Params (M)": j["encoder_params"] / 1e6,
                     "Stage 1 (h)": j.get("stage1_hours"), "Peak GPU (GB)": j.get("peak_gpu_gb"), "metric source": src})
    T = pd.DataFrame(rows); T.round(4).to_csv(d / "R-C.csv", index=False); (d / "R-C.md").write_text(T.round(4).to_markdown(index=False)); (d / "R-C.tex").write_text(T.round(4).to_latex(index=False))
    if 6 in R and 10 in R: notes.append(f"MAD L=6 -> L=10: {R[6]['MAD']:.4f} -> {R[10]['MAD']:.4f} ({100 * (R[10]['MAD'] / R[6]['MAD'] - 1):+.1f}%)")
    lo, hi = min(R), max(R); notes.append(f"MAD L={lo} -> L={hi}: {R[lo]['MAD']:.4f} -> {R[hi]['MAD']:.4f} ({100 * (R[hi]['MAD'] / R[lo]['MAD'] - 1):+.1f}%)")
    pl = {L: j["per_layer_mad"] for L, j in R.items() if isinstance(j.get("per_layer_mad"), dict) and "error" not in j["per_layer_mad"]}
    if pl:
        P = pd.DataFrame([{"model_layers": L, "after_layer": int(k.split("_")[1]), "MAD": v} for L, dd in pl.items() for k, v in dd.items()]); P.to_csv(d / "R-C_per_layer_mad.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for L, g in P.groupby("model_layers"): ax[1].plot(g.after_layer, g.MAD, "-o", ms=3, label=f"L={L}")
        ax[1].set_xlabel("after encoder layer"); ax[1].set_ylabel("MAD"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("MAD by layer")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8)); ax[1].axis("off"); notes.append("no per-layer MAD recorded (run with --per-layer-mad; check the hook error in the json)")
    ax[0].plot(T.Layers, T.MAD, "-o"); ax[0].set_xlabel("encoder layers L"); ax[0].set_ylabel("MAD (final layer, 5,000 nodes)"); ax[0].set_title("MAD vs depth")
    for x in ax:
        if x.axison: x.grid(alpha=.25); x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(d / "fig_mad_vs_depth.png", dpi=600); plt.close(fig)
    for col_, nice in (("Params (M)", "parameters"), ("Stage 1 (h)", "Stage 1 time"), ("Peak GPU (GB)", "peak GPU memory")):
        v = T[col_].astype(float).values
        if np.isnan(v).any(): notes.append(f"PLAUSIBILITY: {nice} missing for some depths; fill it from the logs")
        elif not (np.diff(v) > 0).all(): notes.append(f"PLAUSIBILITY FAIL: {nice} does not rise with L ({dict(zip(T.Layers, v.round(2)))}); re-check the log (different GPU, shared node, early stop) before reporting")
        else: notes.append(f"plausibility ok: {nice} rises with L")
    (d / "R-C_notes.txt").write_text("\n".join(notes) + "\n"); print(T.round(4).to_string(index=False)); print("\n".join(notes))


if __name__ == "__main__":
    main()
