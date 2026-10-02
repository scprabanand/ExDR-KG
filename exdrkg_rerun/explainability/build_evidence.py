"""Step 4: retrieve the evidence paths ONCE per evaluation pair, so every model is checked against the same reference.

For every pair in H1 and H2 the repository's EvidencePathRetriever (max 4 hops, top-10, MMR, lambda 0.3) is run on the
MASKED HKG with the trained encoder/retriever. For each pair the file records
  * `paths`    the retrieved paths as text (what the reasoning engine sees),
  * `entities` every node name on any retrieved path, plus the query drug and disease names. The query names are allowed
               because an explanation must be able to name the drug and disease it is about; all other entities must come
               from a retrieved path to count as supported.
This is the single "evidence" reference used by hallucination_eval.py for ALL models (BioGPT, GPT-4+RAG, TxGNN+template,
V3, V4, V5). The retrieval configuration and checkpoint md5 are written next to the file.

Requires a GPU for the encoder forward pass. Nodes need display names: node_names come from the KG (`raw_ids`); a JSON map
--names (id -> readable name, e.g. DrugBank id -> "Metformin", Entrez id -> "AMPK") is applied when given.

Usage
  python -m exdrkg_rerun.explainability.build_evidence --root . --split standard_degree_matched --masked data/processed/masked_standard.pt \
      --encoder models/hkg_encoder.pt --retriever models/retriever.pt --names data/entity_names.json
Output  results/rerun/hallucination/evidence.jsonl  (+ evidence_config.json)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import pandas as pd, torch
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()
from src.models.hkg_encoder import RelationalGraphTransformer
from src.models.evidence_retriever import EvidencePathRetriever


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--masked", required=True)
    ap.add_argument("--encoder", required=True); ap.add_argument("--retriever", default=None); ap.add_argument("--names", default=None)
    ap.add_argument("--sets", default="results/rerun/hallucination"); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); root = Path(a.root); torch.manual_seed(a.seed); dev = a.device
    kg = torch.load(root / a.masked, weights_only=False, map_location=dev)
    names = json.load(open(a.names)) if a.names else {}
    enc = RelationalGraphTransformer(kg, d_model=256, n_layers=6, n_heads=8, ffn_dim=512, dropout=0.1, device=dev).to(dev); enc.load(root / a.encoder); enc.eval()
    ret = EvidencePathRetriever(enc, kg, max_hops=4, top_k=10, diversity_lambda=0.3, device=dev)
    if a.retriever: ret.load(root / a.retriever)
    ret.eval()
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}; sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    pairs = []
    for s in ("H1_pairs.txt", "H2_pairs.txt"):
        pairs += [l.strip() for l in open(root / a.sets / s) if l.strip()]
    pairs = sorted(set(pairs)); out = root / a.sets / "evidence.jsonl"; t0 = time.time(); n_no_path = 0
    nm = lambda x: names.get(str(x), str(x))
    with open(out, "w") as f:
        for p in pairs:
            d, s = p.split("|")
            with torch.no_grad(): paths, _ = ret.retrieve(did[d], sid[s])
            ents = {nm(d).lower(), nm(s).lower()}
            for pa in paths: ents |= {nm(x).lower() for x in pa.node_names}
            n_no_path += int(len(paths) == 0)
            f.write(json.dumps({"pair_id": p, "drug": nm(d), "disease": nm(s), "paths": [pa.to_nl() for pa in paths], "entities": sorted(ents), "n_paths": len(paths)}) + "\n")
    json.dump({"pairs": len(pairs), "pairs_without_any_path": n_no_path, "max_hops": 4, "top_k": 10, "diversity_lambda": 0.3, "seed": a.seed,
               "encoder_md5": md5(root / a.encoder), "retriever_md5": md5(root / a.retriever) if a.retriever else None, "masked_graph": a.masked,
               "seconds": time.time() - t0}, open(root / a.sets / "evidence_config.json", "w"), indent=1)
    print(f"evidence for {len(pairs)} pairs ({n_no_path} without any path) -> {out}")


if __name__ == "__main__":
    main()
