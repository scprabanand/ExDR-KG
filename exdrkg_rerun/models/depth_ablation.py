"""Step 9 (Comment 1.4): encoder depth ablation, L = 2, 4, 8, 10 trained here; L = 6 taken from the EXISTING checkpoint.

Everything except the number of layers is identical for every depth: 8 heads, d_model 256, ffn 512, dropout 0.1, mask ratio 0.15,
the repository's three pre-training objectives (RelationalGraphTransformer.pretrain_step), 100 epochs, lr 1e-4, seed 42, the same
masked HKG and the same rebuilt split/negative pool (--split). The encoder is then frozen and the V1 MLP link predictor is trained on
the frozen drug/disease embeddings (same head, same seed, model selection on VALIDATION AUPRC only) and evaluated on the held-out test set.

Recorded per depth (results/rerun/<split>/depth_L<L>.json)
  metrics + 95% CIs (shared metrics.py), MAD of final-layer embeddings (5,000 random nodes, seed 42, the SAME node sample for every
  depth; the sample's md5 is stored so this can be verified), encoder parameter count, Stage-1 wall-clock hours, peak GPU memory,
  and (--per-layer-mad) the MAD after every encoder layer, from forward hooks on model.layers.
L = 6 (do NOT retrain):
    python -m exdrkg_rerun.models.depth_ablation --root . --split standard_degree_matched --layers 6 --checkpoint checkpoints/hkg_encoder.pt \\
        --stage1-hours <from the Stage-1 training log> --peak-gpu-gb <from the log> --per-layer-mad
  This loads the checkpoint, skips Stage 1, recomputes MAD, and trains the same V1 head. depth_summary.py then takes the L = 6 metrics
  from the V1 scores file of Step 2/3 (so they equal V1 in Table R-B by construction) and reports how far the head re-run here is from it.
Other depths (one job each; ~18 h at L=6 in the manuscript setup, more for L=8/10):
    python -m exdrkg_rerun.models.depth_ablation --root . --split standard_degree_matched --layers 2
Timing note: stage1_hours is measured here from a fresh start of the training loop only (no head training, no evaluation).
"""
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from exdrkg_rerun.data.masked_kg import build_masked_kg
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci
from src.models.hkg_encoder import RelationalGraphTransformer


def mad_indices(n_total, n=5000, seed=42):
    torch.manual_seed(seed); return torch.randperm(n_total)[:n]       # exactly the snippet of the plan; identical for every depth


def mad(H: torch.Tensor, idx: torch.Tensor) -> float:
    Z = F.normalize(H[idx].float().cpu(), dim=1); D = 1 - Z @ Z.T                     # cosine distance matrix
    return D[~torch.eye(len(idx), dtype=torch.bool)].mean().item()


def flatten(out, node_types):
    """Turn a layer output (Tensor | dict type->Tensor | tuple/list) into one [N, d] matrix in node_types order."""
    if isinstance(out, (tuple, list)): out = out[0]
    if isinstance(out, dict): return torch.cat([out[t] for t in node_types], 0)
    return out


class LinkMLP(nn.Module):  # V1 link predictor: concat(drug, disease) -> MLP (hidden 512, dropout 0.1)
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 1))
    def forward(self, a, b): return self.f(torch.cat([a, b], 1)).squeeze(-1)


def per_layer_mad(model, idx):
    layers = getattr(model, "layers", None)
    if layers is None: return {"error": "model has no attribute 'layers'; add hooks for the repository's layer container"}
    store, hooks = [], []
    for i, m in enumerate(layers): hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: store.append((i + 1, flatten(out, model.node_types).detach()))))
    with torch.no_grad(): model()
    for h in hooks: h.remove()
    if len(store) != len(layers): return {"error": f"hooks fired {len(store)} times for {len(layers)} layers"}
    return {f"layer_{i}": mad(H, idx) for i, H in store}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard_degree_matched")
    ap.add_argument("--layers", type=int, required=True); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--head-epochs", type=int, default=50); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-layer-mad", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None, help="existing encoder checkpoint: skips Stage 1 (use for L=6, never retrain it)")
    ap.add_argument("--stage1-hours", type=float, default=None, help="with --checkpoint: Stage-1 hours copied from the original log")
    ap.add_argument("--peak-gpu-gb", type=float, default=None, help="with --checkpoint: peak GPU memory (GB) copied from the original log")
    ap.add_argument("--save-encoder", default=None, help="where to save the trained encoder (default checkpoints/rerun/encoder_L<L>_<split>.pt)")
    a = ap.parse_args(); root = Path(a.root); dev = a.device
    if a.layers == 6 and not a.checkpoint: raise SystemExit("L = 6 must not be retrained: pass --checkpoint <existing encoder> (plus --stage1-hours/--peak-gpu-gb from its log)")
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    kg, info = build_masked_kg(root, a.split)
    model = RelationalGraphTransformer(kg, d_model=256, n_layers=a.layers, n_heads=8, ffn_dim=512, dropout=0.1, mask_ratio=0.15, device=dev).to(dev)
    n_params = sum(p.numel() for p in model.parameters()); res = {"layers": a.layers, "split": a.split, "seed": a.seed, "encoder_params": n_params}
    if a.checkpoint:
        sd = torch.load(root / a.checkpoint, map_location=dev, weights_only=False)
        sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=True)            # strict: a config mismatch (layers, d_model, heads) must fail loudly
        res.update({"stage1_hours": a.stage1_hours, "peak_gpu_gb": a.peak_gpu_gb, "stage1": "existing checkpoint (not retrained); time and memory copied from its log", "checkpoint": a.checkpoint})
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        for ep in range(a.epochs):                    # Stage 1 (unsupervised, no labels used)
            model.train(); opt.zero_grad(); loss = model.pretrain_step(mask_ratio=0.15)["total"]; loss.backward(); opt.step()
            if ep % 10 == 0: print(f"L={a.layers} epoch {ep} loss {loss.item():.4f} t={(time.time() - t0) / 3600:.2f}h", flush=True)
        torch.cuda.synchronize(); res.update({"stage1_hours": (time.time() - t0) / 3600, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "epochs": a.epochs, "stage1": "trained here"})
        ck = Path(a.save_encoder or root / "checkpoints" / "rerun" / f"encoder_L{a.layers}_{a.split}.pt"); ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "layers": a.layers, "seed": a.seed}, ck); res["checkpoint"] = str(ck)
    model.eval()
    with torch.no_grad(): h = model()
    H = torch.cat([h[t] for t in model.node_types], 0); idx = mad_indices(H.size(0), 5000, a.seed)
    res.update({"n_nodes_total": int(H.size(0)), "MAD": mad(H, idx), "mad_sample_md5": hashlib.md5(idx.numpy().tobytes()).hexdigest(), "mad_sample_size": int(len(idx))})
    if a.per_layer_mad: res["per_layer_mad"] = per_layer_mad(model, idx)
    # V1 head on frozen embeddings (same head, same seed for every depth)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}; sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    def tens(df): return (torch.tensor(df.drug_id.map(did).values, device=dev), torch.tensor(df.disease_id.map(sid).values, device=dev),
                          torch.tensor(df.label.values, dtype=torch.float32, device=dev))
    tr, va, te = [pd.read_csv(root / f"data/splits/{a.split}/{k}.csv") for k in ("train", "val", "test")]
    head = LinkMLP(256).to(dev); o2 = torch.optim.Adam(head.parameters(), lr=1e-4)
    d_tr, s_tr, y_tr = tens(tr); best, best_state = -1, None
    pos_w = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), device=dev)
    from sklearn.metrics import average_precision_score
    h = {k: v.detach() for k, v in h.items()}
    for ep in range(a.head_epochs):
        head.train(); perm = torch.randperm(len(y_tr), device=dev)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; o2.zero_grad()
            F.binary_cross_entropy_with_logits(head(h["drug"][d_tr[b]], h["disease"][s_tr[b]]), y_tr[b], pos_weight=pos_w).backward(); o2.step()
        head.eval()
        with torch.no_grad():
            dv, sv, yv = tens(va); pv = torch.sigmoid(head(h["drug"][dv], h["disease"][sv])).cpu().numpy()
        ap_v = average_precision_score(yv.cpu().numpy(), pv)      # model selection on VALIDATION only
        if ap_v > best: best, best_state = ap_v, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval(); res["head_params"] = sum(p.numel() for p in head.parameters()); res["best_val_auprc"] = float(best)
    with torch.no_grad():
        dt, st, yt = tens(te); sc = torch.sigmoid(head(h["drug"][dt], h["disease"][st])).cpu().numpy()
    res["metrics"] = compute_metrics(te, sc); res["ci95"] = bootstrap_ci(te, sc)
    out = root / "results" / "rerun" / a.split; (out / "depth_scores").mkdir(parents=True, exist_ok=True)
    te.assign(score=sc)[["drug_id", "disease_id", "label", "score"]].to_csv(out / "depth_scores" / f"L{a.layers}.csv", index=False)
    json.dump(res, open(out / f"depth_L{a.layers}.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
