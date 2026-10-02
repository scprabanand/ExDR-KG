"""Step 2 (model 8): score every test pair with a trained ExDR-KG checkpoint (scoring pass only, Eq. 22).

No explanation generation is performed. The graph passed to the model MUST be the masked HKG built by masked_kg.py
(the shipped unified_kg.pt still contains the held-out test edges). Works for V1-V5 checkpoints so the same script
feeds Step 3 (`--tag V1` ... `--tag V5`); the ExDR-KG row of Table R-A must equal V5.

Usage
  python -m exdrkg_rerun.models.score_exdrkg --root . --split standard_degree_matched --masked data/processed/masked_standard.pt \
      --checkpoint models/exdr_kg_v5.pt --tag ExDR-KG --device cuda
Output  results/rerun/main/scores/<tag>.csv  (drug_id, disease_id, label, score) + <tag>.meta.json
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd, torch
from src.models import ExDRKG, ExDRKGConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--masked", required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--tag", default="ExDR-KG"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); root = Path(a.root); torch.manual_seed(a.seed)
    from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()
    kg = torch.load(root / a.masked, weights_only=False, map_location=a.device)
    cfg = ExDRKGConfig.from_yaml(root / a.config)
    model = ExDRKG(cfg, kg_data=kg, device=a.device); model.load(root / a.checkpoint); model.eval()
    test = pd.read_csv(root / "data/splits" / a.split / "test.csv")
    pairs = list(zip(test.drug_id, test.disease_id)); scores = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pairs), a.batch_size):
            res = model.infer_batch(pairs[i:i + a.batch_size], batch_size=a.batch_size)
            scores += [float(r["score"]) for r in res]
    assert len(scores) == len(test), "every test pair must receive a score"
    out = root / "results/rerun/main/scores"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"drug_id": test.drug_id, "disease_id": test.disease_id, "label": test.label, "score": scores}).to_csv(out / f"{a.tag}.csv", index=False)
    json.dump({"checkpoint": a.checkpoint, "masked_graph": a.masked, "split": a.split, "seed": a.seed, "seconds": time.time() - t0,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, open(out / f"{a.tag}.meta.json", "w"), indent=1)
    print("scored", len(scores), "pairs ->", out / f"{a.tag}.csv")


if __name__ == "__main__":
    main()
