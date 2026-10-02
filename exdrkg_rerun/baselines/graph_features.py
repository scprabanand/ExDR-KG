"""Leakage-safe graph features for drug-disease pairs computed on the unified HKG.

The base graph EXCLUDES every drug-disease indication edge (treats / palliates and reverses), so the
held-out labels can never be recovered from the structure. Known training indications enter only through
`indication_features`, and for pairs in the training set the pair's own fold of positives is hidden
(K-fold masking) so a training positive never sees its own edge.
"""
from __future__ import annotations
import numpy as np, scipy.sparse as sp, torch, warnings
warnings.filterwarnings("ignore")
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

INDICATION_REL = {"treats", "rev_treats", "palliates", "rev_palliates"}

class HKG:
    def __init__(self, kg_path):
        kg = torch.load(kg_path, weights_only=False)
        self.types = kg.node_types; self.off = {}; o = 0
        for t in self.types: self.off[t] = o; o += kg[t].num_nodes
        self.N = o
        self.node_type = np.concatenate([[i] * kg[t].num_nodes for i, t in enumerate(self.types)])
        self.drug_ids = list(kg["drug"].raw_ids); self.dis_ids = list(kg["disease"].raw_ids)
        self.did = {r: i for i, r in enumerate(self.drug_ids)}; self.sid = {r: i for i, r in enumerate(self.dis_ids)}
        rows, cols = [], []
        self.indications = []                       # all (drug_idx, disease_idx) indication edges in the shipped KG
        for et in kg.edge_types:
            s, r, d = et; ei = kg[et].edge_index.numpy()
            if r in INDICATION_REL:
                if s == "drug": self.indications += list(zip(ei[0].tolist(), ei[1].tolist()))
                continue
            rows.append(ei[0] + self.off[s]); cols.append(ei[1] + self.off[d])
        r = np.concatenate(rows); c = np.concatenate(cols)
        A = sp.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(self.N, self.N)).tocsr()
        A = ((A + A.T) > 0).astype(np.float32).tocsr(); A.setdiag(0); A.eliminate_zeros()
        self.A = A; self.deg = np.asarray(A.sum(1)).ravel()
        self.nd, self.ns = len(self.drug_ids), len(self.dis_ids)
        self.drug_rows = np.arange(self.nd) + self.off["drug"]; self.dis_cols = np.arange(self.ns) + self.off["disease"]

    def base_feature_matrices(self):
        """Return dict name -> (n_drugs x n_diseases) matrices of path statistics."""
        A = self.A; deg = self.deg
        w = (1.0 / np.log(2.0 + deg)).astype(np.float32)
        Ad = A[self.drug_rows]                              # drugs x N (walks of length 1)
        Acol = A[:, self.dis_cols].tocsr()                  # N x diseases
        Wd = sp.diags(w)
        out = {}
        W1 = Ad @ Acol                                      # never non-zero here (no direct edge): kept for completeness
        out["walk1"] = W1.toarray()
        # length 2 walks, total, Adamic-Adar weighted, and by intermediate node type
        # length-2: drug - x - disease
        out["walk2"] = (Ad @ Acol).toarray()
        out["aa2"] = ((Ad @ Wd) @ Acol).toarray()
        for ti, t in enumerate(self.types):
            m = sp.diags((self.node_type == ti).astype(np.float32))
            X = ((Ad @ m) @ Acol).toarray()
            if X.sum() > 0: out[f"walk2_via_{t}"] = X
        # length-3 and length-4 walks (dense drug x N intermediate; chunked for memory)
        W3 = np.zeros((self.nd, self.ns), np.float32); W3w = np.zeros_like(W3); W4 = np.zeros_like(W3)
        B2 = (A @ Acol).tocsr()                             # N x diseases, length-2 walks to each disease
        for a in range(0, self.nd, 400):
            blk = Ad[a:a + 400]
            S2 = (blk @ A).tocsr()                          # walks of length 2 from these drugs to all nodes
            W3[a:a + 400] = (S2 @ Acol).toarray()
            W3w[a:a + 400] = ((S2 @ Wd) @ Acol).toarray()
            W4[a:a + 400] = (S2 @ B2).toarray()
        out["walk3"], out["aa3"], out["walk4"] = W3, W3w, W4
        # shortest distance (min k with a walk of length k); 5 = not reachable within 4 hops
        dist = np.full((self.nd, self.ns), 5.0, np.float32)
        for k, key in [(4, "walk4"), (3, "walk3"), (2, "walk2")]:
            dist[out[key] > 0] = k
        out["dist"] = dist
        # gene-profile overlap (drug-gene vs disease-gene neighbour sets)
        gi = self.types.index("gene"); gmask = (self.node_type == gi)
        Dg = (Ad[:, gmask] > 0).astype(np.float32); Sg = (A[self.dis_cols][:, gmask] > 0).astype(np.float32)
        inter = (Dg @ Sg.T).toarray(); dn = np.asarray(Dg.sum(1)).ravel()[:, None]; sn = np.asarray(Sg.sum(1)).ravel()[None, :]
        out["gene_inter"] = inter; out["gene_jacc"] = inter / np.maximum(dn + sn - inter, 1)
        out["gene_cos"] = inter / np.maximum(np.sqrt(dn * sn), 1)
        # gene-neighbourhood similarity between drugs and between diseases (for collaborative features)
        self.Dg = Dg.tocsr(); self.Sg = Sg.tocsr()
        di = (Dg @ Dg.T).toarray(); dd = np.asarray(Dg.sum(1)).ravel()
        self.drug_sim = di / np.maximum(dd[:, None] + dd[None, :] - di, 1)
        si = (Sg @ Sg.T).toarray(); ss = np.asarray(Sg.sum(1)).ravel()
        self.dis_sim = si / np.maximum(ss[:, None] + ss[None, :] - si, 1)
        np.fill_diagonal(self.drug_sim, 0); np.fill_diagonal(self.dis_sim, 0)
        out["drug_deg"] = np.repeat(np.log1p(deg[self.drug_rows])[:, None], self.ns, 1)
        out["dis_deg"] = np.repeat(np.log1p(deg[self.dis_cols])[None, :], self.nd, 0)
        return out

    def indication_features(self, T):
        """T: (nd x ns) 0/1 matrix of KNOWN indications visible to the model. Collaborative features."""
        T = T.astype(np.float32)
        # sum/max similarity of drug to the drugs already known to treat the disease
        f_drug_sum = self.drug_sim @ T                                   # nd x ns
        cnt = T.sum(0, keepdims=True)
        f_drug_mean = f_drug_sum / np.maximum(cnt, 1)
        f_drug_max = np.zeros_like(T)
        for s in range(self.ns):
            nz = np.where(T[:, s] > 0)[0]
            if len(nz): f_drug_max[:, s] = self.drug_sim[:, nz].max(1)
        # similarity of disease to diseases the drug is known to treat
        f_dis_sum = T @ self.dis_sim
        f_dis_max = np.zeros_like(T)
        for d in range(self.nd):
            nz = np.where(T[d] > 0)[0]
            if len(nz): f_dis_max[d] = self.dis_sim[nz].max(0)
        return {"ind_drug_sum": f_drug_sum, "ind_drug_mean": f_drug_mean, "ind_drug_max": f_drug_max,
                "ind_dis_sum": f_dis_sum, "ind_dis_max": f_dis_max,
                "ind_drug_ntreat": np.repeat(np.log1p(T.sum(1))[:, None], self.ns, 1),
                "ind_dis_ntreat": np.repeat(np.log1p(T.sum(0))[None, :], self.nd, 0)}
