"""Step 1 helper: build the leakage-safe (masked) HKG for a split and record file hashes.

The shipped data/processed/unified_kg.pt still contains EVERY drug-disease indication edge, including the
validation and test positives. The encoder pre-trains with link prediction on all edges and message-passes over
them, so it must never be given that file directly. This script removes, for every validation and test positive
(d, q): the treats/palliates edge, its reverse, and any parallel duplicate, and (for the temporal setting) can
also drop edges after a cutoff if an edge-timestamp table is supplied.

Usage:
    python -m exdrkg_rerun.data.masked_kg --root . --split standard --out data/processed/masked_standard.pt
"""
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import pandas as pd, torch
from torch_geometric.data import HeteroData
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

INDICATION = {"treats", "palliates"}
REV = {"rev_treats", "rev_palliates"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_masked_kg(root: Path, split: str) -> tuple[HeteroData, dict]:
    kg = torch.load(root / "data/processed/unified_kg.pt", weights_only=False)
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}
    sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    held = set()
    for part in ("val", "test"):
        p = root / f"data/splits/{split}/{part}.csv"
        if p.exists():
            d = pd.read_csv(p)
            held |= {(did[a], sid[b]) for a, b in zip(d[d.label == 1].drug_id, d[d.label == 1].disease_id)}
    removed = 0
    for et in kg.edge_types:
        s, r, t = et
        if r in INDICATION or r in REV:
            ei = kg[et].edge_index
            pairs = zip(ei[0].tolist(), ei[1].tolist()) if s == "drug" else zip(ei[1].tolist(), ei[0].tolist())
            keep = torch.tensor([(p not in held) for p in pairs], dtype=torch.bool)
            removed += int((~keep).sum())
            kg[et].edge_index = ei[:, keep]
    info = {"split": split, "held_out_positive_pairs": len(held), "directed_edges_removed": removed}
    # verification: no held-out pair may remain in ANY drug-disease edge type
    for et in kg.edge_types:
        s, r, t = et
        if {s, t} == {"drug", "disease"}:
            ei = kg[et].edge_index
            pairs = zip(ei[0].tolist(), ei[1].tolist()) if s == "drug" else zip(ei[1].tolist(), ei[0].tolist())
            assert not any(p in held for p in pairs), f"leak in {et}"
    return kg, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); root = Path(a.root)
    kg, info = build_masked_kg(root, a.split)
    out = Path(a.out or root / f"data/processed/masked_{a.split}.pt"); torch.save(kg, out)
    files = [root / "data/processed/unified_kg.pt", out] + sorted((root / "data/splits").glob("*/*.csv"))
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    facts = {"commit": commit, "python": platform.python_version(), "torch": torch.__version__,
             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             "masking": info, "md5": {(str(f.relative_to(root)) if root.resolve() in f.resolve().parents else f.name): md5(f) for f in files}}
    (root / "results").mkdir(exist_ok=True)
    json.dump(facts, open(root / "results" / f"R-K_freeze_{a.split}.json", "w"), indent=1)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
