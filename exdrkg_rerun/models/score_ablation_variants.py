"""Step 3: score the SAME test pairs with every saved ablation checkpoint (V1-V5). No retraining, no explanation generation.

The variants differ in which modules are active and in how the score is produced, so each is described in a small
manifest (see ablation_manifest.example.yaml) instead of being hard-coded:

  scorer: encoder_mlp   V1. RelationalGraphTransformer + LinkMLP (the V1 head of depth_ablation.py); needs encoder and head state dicts.
  scorer: exdrkg        V2-V5. ExDRKG built from configs/ablation_study.yaml with the variant's module switches
                        (`config_key`), then `ExDRKG.load(checkpoint)`; score = model score (Eq. 22).

Every variant runs on the MASKED graph (masked_kg.py), the same test file, batch size and seed. Outputs are
results/rerun/ablation/scores/<V>.csv and <V>.meta.json (checkpoint md5, seconds, GPU). V5 is also copied to
results/rerun/main/scores/ExDR-KG.csv unless --no-copy-v5, so that Table R-A and Table R-B share the very same file.

Usage
  python -m exdrkg_rerun.models.score_ablation_variants --root . --split standard_degree_matched --masked data/processed/masked_standard.pt \
      --manifest rerun/configs/ablation_manifest.yaml --device cuda
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import pandas as pd, torch, torch.nn as nn, yaml
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


class LinkMLP(nn.Module):   # identical to depth_ablation.py
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 1))
    def forward(self, a, b): return self.f(torch.cat([a, b], 1)).squeeze(-1)


def score_encoder_mlp(spec, kg, test, dev):
    from src.models.hkg_encoder import RelationalGraphTransformer
    enc = RelationalGraphTransformer(kg, d_model=256, n_layers=spec.get("n_layers", 6), n_heads=8, ffn_dim=512, dropout=0.1, device=dev).to(dev)
    ck = torch.load(spec["checkpoint"], map_location=dev)
    enc.load_state_dict(ck["encoder"]); head = LinkMLP(256).to(dev); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
    did = {r: i for i, r in enumerate(kg["drug"].raw_ids)}; sid = {r: i for i, r in enumerate(kg["disease"].raw_ids)}
    d = torch.tensor(test.drug_id.map(did).values, device=dev); s = torch.tensor(test.disease_id.map(sid).values, device=dev)
    with torch.no_grad():
        h = enc(); return torch.sigmoid(head(h["drug"][d], h["disease"][s])).cpu().tolist()


def score_exdrkg(spec, kg, test, dev, bs, cfg_path):
    from src.models import ExDRKG, ExDRKGConfig
    cfg_dict = yaml.safe_load(open(cfg_path)); variant = yaml.safe_load(open(spec["ablation_config"]))["ablation_study"][spec["config_key"]]
    for mod, sw in variant.get("modules", {}).items():            # apply the variant's module switches on top of the default config
        cfg_dict.setdefault("model", {}).setdefault(mod, {}).update(sw)
    model = ExDRKG(ExDRKGConfig(cfg_dict), kg_data=kg, device=dev); model.load(spec["checkpoint"]); model.eval()
    pairs = list(zip(test.drug_id, test.disease_id)); out = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            out += [float(r["score"]) for r in model.infer_batch(pairs[i:i + bs], batch_size=bs)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard"); ap.add_argument("--masked", required=True)
    ap.add_argument("--manifest", default="rerun/configs/ablation_manifest.yaml"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--no-copy-v5", action="store_true")
    a = ap.parse_args(); root = Path(a.root); torch.manual_seed(a.seed)
    man = yaml.safe_load(open(root / a.manifest))["variants"]
    kg = torch.load(root / a.masked, weights_only=False, map_location=a.device)
    test = pd.read_csv(root / "data/splits" / a.split / "test.csv"); out = root / "results/rerun/ablation/scores"; out.mkdir(parents=True, exist_ok=True)
    for v in (a.only or list(man)):
        spec = dict(man[v]); spec["checkpoint"] = str(root / spec["checkpoint"]); t0 = time.time()
        if "ablation_config" in spec: spec["ablation_config"] = str(root / spec["ablation_config"])
        sc = score_encoder_mlp(spec, kg, test, a.device) if spec["scorer"] == "encoder_mlp" else score_exdrkg(spec, kg, test, a.device, a.batch_size, root / a.config)
        assert len(sc) == len(test), f"{v}: every test pair must be scored"
        pd.DataFrame({"drug_id": test.drug_id, "disease_id": test.disease_id, "label": test.label, "score": sc}).to_csv(out / f"{v}.csv", index=False)
        json.dump({"variant": v, "checkpoint": spec["checkpoint"], "checkpoint_md5": md5(spec["checkpoint"]), "scorer": spec["scorer"], "seed": a.seed,
                   "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, open(out / f"{v}.meta.json", "w"), indent=1)
        print(v, "scored", len(sc), "pairs")
    if not a.no_copy_v5 and (out / "V5.csv").exists():
        d = root / "results/rerun/main/scores"; d.mkdir(parents=True, exist_ok=True); shutil.copy(out / "V5.csv", d / "ExDR-KG.csv"); print("copied V5 -> Table R-A ExDR-KG row")


if __name__ == "__main__":
    main()
