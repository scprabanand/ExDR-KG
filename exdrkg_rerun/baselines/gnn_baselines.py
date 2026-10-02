"""Step 2 (models 1-4): train and score Metapath2vec, HAN, R-GCN and CompGCN on the SAME masked HKG, splits and negatives.

Protocol shared by every baseline (so differences reflect architecture, not data):
  * graph: masked HKG from masked_kg.py (validation/test indication edges, reverses and duplicates removed);
  * supervision: train pairs of the chosen split (BCE, positives up-weighted by the negative:positive ratio);
  * model selection: validation AUPRC only (test pairs are never used for early stopping or tuning);
  * head: concat(drug embedding, disease embedding) -> MLP(512) -> logit, the same head as V1;
  * scoring: all test pairs -> results/rerun/main/scores/<Model>.csv; seed 42; embedding size 256 (Metapath2vec 128).
Hyper-parameters are the ones in Table 4 of the manuscript unless the original paper's defaults are stated below.

Usage
  python -m exdrkg_rerun.baselines.gnn_baselines --root . --split standard_degree_matched --masked data/processed/masked_standard.pt --model rgcn
  (--model in {rgcn, compgcn, han, metapath2vec}); one run per model, roughly 1-4 h each on one A100.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import RGCNConv, HANConv, MetaPath2Vec
from torch_geometric.utils import scatter
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

D = 256


class Head(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 1))
    def forward(self, a, b): return self.f(torch.cat([a, b], 1)).squeeze(-1)


def flatten(kg, dev):
    """Homogeneous view: global node ids, per-edge relation id, per-type input projection."""
    off, o = {}, 0
    for t in kg.node_types: off[t] = o; o += kg[t].num_nodes
    src, dst, rel = [], [], []
    for r_id, et in enumerate(kg.edge_types):
        s, _, d = et; ei = kg[et].edge_index
        src.append(ei[0] + off[s]); dst.append(ei[1] + off[d]); rel.append(torch.full((ei.size(1),), r_id))
    ei = torch.stack([torch.cat(src), torch.cat(dst)]).to(dev); et = torch.cat(rel).to(dev)
    return off, o, ei, et, len(kg.edge_types)


class Inputs(nn.Module):
    def __init__(self, kg):
        super().__init__(); self.types = list(kg.node_types)
        self.proj = nn.ModuleDict({t: nn.Linear(kg[t].x.size(1), D) for t in self.types}); self.kg = kg
    def forward(self):
        return {t: self.proj[t](self.kg[t].x.float()) for t in self.types}


class RGCN(nn.Module):
    def __init__(self, kg, n_rel, dev):
        super().__init__(); self.inp = Inputs(kg); self.off, self.N, self.ei, self.et, _ = flatten(kg, dev)
        self.c1 = RGCNConv(D, D, n_rel, num_bases=30); self.c2 = RGCNConv(D, D, n_rel, num_bases=30); self.types = self.inp.types; self.kg = kg
    def forward(self):
        h = self.inp(); x = torch.cat([h[t] for t in self.types])
        x = F.relu(self.c1(x, self.ei, self.et)); x = self.c2(x, self.ei, self.et)
        return {t: x[self.off[t]: self.off[t] + self.kg[t].num_nodes] for t in self.types}


class CompGCNLayer(nn.Module):
    """CompGCN (Vashishth et al. 2020), composition = element-wise multiplication, one shared transform per layer."""
    def __init__(self, n_rel):
        super().__init__(); self.rel = nn.Parameter(torch.randn(n_rel, D) * 0.1); self.w = nn.Linear(D, D); self.w_r = nn.Linear(D, D)
    def forward(self, x, ei, et):
        msg = self.w(x[ei[0]] * self.rel[et]); agg = scatter(msg, ei[1], dim=0, dim_size=x.size(0), reduce="mean")
        self.rel_out = self.w_r(self.rel); return F.relu(agg + x)


class CompGCN(nn.Module):
    def __init__(self, kg, n_rel, dev):
        super().__init__(); self.inp = Inputs(kg); self.off, self.N, self.ei, self.et, _ = flatten(kg, dev)
        self.l1 = CompGCNLayer(n_rel); self.l2 = CompGCNLayer(n_rel); self.types = self.inp.types; self.kg = kg
    def forward(self):
        h = self.inp(); x = torch.cat([h[t] for t in self.types]); x = self.l2(self.l1(x, self.ei, self.et), self.ei, self.et)
        return {t: x[self.off[t]: self.off[t] + self.kg[t].num_nodes] for t in self.types}


class HAN(nn.Module):
    def __init__(self, kg):
        super().__init__(); self.inp = Inputs(kg); md = kg.metadata()
        self.c1 = HANConv(D, D, md, heads=8, dropout=0.1); self.c2 = HANConv(D, D, md, heads=8, dropout=0.1); self.kg = kg
    def forward(self):
        h = self.inp(); ei = {et: self.kg[et].edge_index for et in self.kg.edge_types}
        h = self.c1(h, ei); h = {t: F.relu(v) if v is not None else self.inp()[t] for t, v in h.items()}
        h = self.c2(h, ei); return {t: (v if v is not None else torch.zeros(self.kg[t].num_nodes, D, device=v.device if v is not None else 'cuda')) for t, v in h.items()}


def train_metapath2vec(kg, dev, seed):
    """Heterogeneous random-walk embeddings (Dong et al. 2017). Metapath: drug-gene-disease-gene-drug and the reverse leg."""
    mp = [("drug", "binds", "gene"), ("gene", "rev_associates", "disease"), ("disease", "associates", "gene"), ("gene", "rev_binds", "drug")]
    m = MetaPath2Vec(kg.edge_index_dict, embedding_dim=128, metapath=mp, walk_length=50, context_size=7, walks_per_node=5, num_negative_samples=5, sparse=True).to(dev)
    loader = m.loader(batch_size=256, shuffle=True, num_workers=4); opt = torch.optim.SparseAdam(list(m.parameters()), lr=0.01)
    for ep in range(5):
        for pos, neg in loader:
            opt.zero_grad(); m.loss(pos.to(dev), neg.to(dev)).backward(); opt.step()
    return {t: m(t).detach() for t in ("drug", "disease")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--masked", required=True)
    ap.add_argument("--model", choices=["rgcn", "compgcn", "han", "metapath2vec"], required=True)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); root = Path(a.root); dev = a.device; torch.manual_seed(a.seed); np.random.seed(a.seed)
    kg = torch.load(root / a.masked, weights_only=False).to(dev); n_rel = len(kg.edge_types)
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}; sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    sp = {k: pd.read_csv(root / "data/splits" / a.split / f"{k}.csv") for k in ("train", "val", "test")}
    T = lambda df: (torch.tensor(df.drug_id.map(did).values, device=dev), torch.tensor(df.disease_id.map(sid).values, device=dev), torch.tensor(df.label.values, dtype=torch.float32, device=dev))
    (dt, st, yt), (dv, sv, yv), (de, se, ye) = T(sp["train"]), T(sp["val"]), T(sp["test"])
    t0 = time.time()
    if a.model == "metapath2vec":
        emb = train_metapath2vec(kg, dev, a.seed); enc = None; head = Head(128).to(dev); params = list(head.parameters())
    else:
        enc = {"rgcn": lambda: RGCN(kg, n_rel, dev), "compgcn": lambda: CompGCN(kg, n_rel, dev), "han": lambda: HAN(kg)}[a.model]().to(dev)
        head = Head(D).to(dev); params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-5); pw = (yt == 0).sum() / (yt == 1).sum()
    best, state = -1.0, None
    for ep in range(a.epochs):
        if enc is not None: enc.train()
        head.train(); opt.zero_grad()
        h = emb if enc is None else enc()
        idx = torch.randint(0, len(yt), (4096,), device=dev)          # mini-batch of labelled pairs, full graph pass
        loss = F.binary_cross_entropy_with_logits(head(h["drug"][dt[idx]], h["disease"][st[idx]]), yt[idx], pos_weight=pw); loss.backward(); opt.step()
        if ep % 5 == 4:
            if enc is not None: enc.eval()
            head.eval()
            with torch.no_grad():
                h = emb if enc is None else enc(); pv = torch.sigmoid(head(h["drug"][dv], h["disease"][sv])).cpu().numpy()
            ap_v = average_precision_score(yv.cpu().numpy(), pv)      # model selection on VALIDATION only
            if ap_v > best: best = ap_v; state = ({k: v.clone() for k, v in enc.state_dict().items()} if enc is not None else None, {k: v.clone() for k, v in head.state_dict().items()})
    if enc is not None: enc.load_state_dict(state[0]); enc.eval()
    head.load_state_dict(state[1]); head.eval()
    with torch.no_grad():
        h = emb if enc is None else enc(); sc = torch.sigmoid(head(h["drug"][de], h["disease"][se])).cpu().numpy()
    name = {"rgcn": "R-GCN", "compgcn": "CompGCN", "han": "HAN", "metapath2vec": "Metapath2vec"}[a.model]
    out = root / "results/rerun/main/scores"; out.mkdir(parents=True, exist_ok=True)
    te = sp["test"]; pd.DataFrame({"drug_id": te.drug_id, "disease_id": te.disease_id, "label": te.label, "score": sc}).to_csv(out / f"{name}.csv", index=False)
    json.dump({"model": name, "best_val_AUPRC": best, "hours": (time.time() - t0) / 3600, "seed": a.seed, "split": a.split}, open(out / f"{name}.meta.json", "w"), indent=1)
    print(name, "best val AUPRC", round(best, 4), "-> scored", len(sc), "test pairs")


if __name__ == "__main__":
    main()
