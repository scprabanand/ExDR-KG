"""Step 3 side question: does PPO (Stage 3) update the same LoRA weights that produce the repurposing score s_dq (Eq. 15)?

Answering "Yes" or "No" from the code alone is not reliable, so this script measures it from the saved checkpoints. It
compares the V4 adapter (end of Stage 2) with the V5 adapter (end of Stage 3) tensor by tensor:

  * if PPO changed LoRA tensors that are also used in the scoring forward pass (all LoRA A/B matrices are), the answer is
    YES: PPO moves the score, which is why V5 differs from V4 in AUROC;
  * if every LoRA and score-head tensor is bit-identical between V4 and V5, the answer is NO: PPO trained a separate
    module (for example a separate explainer/reward model), so any V4->V5 AUROC change would have to come from elsewhere
    (for example a different candidate set or re-scoring), which the manuscript would need to explain.

It also reports which parameters exist only in V5 (reward model, value head, separate adapter) so the answer can be stated
precisely. Works with PEFT adapter folders (adapter_model.safetensors) or plain torch state dicts.

Usage
  python -m exdrkg_rerun.models.check_ppo_lora_sharing --v4 models/ablation/v4 --v5 models/ablation/v5 --out results/rerun/ablation/ppo_lora_check.json
"""
import argparse, json
from pathlib import Path
import torch


def load_state(p: Path) -> dict:
    p = Path(p)
    if p.is_dir():
        f = p / "adapter_model.safetensors"
        if f.exists():
            from safetensors.torch import load_file; state = load_file(str(f))
        else:
            state = torch.load(p / "adapter_model.bin", map_location="cpu")
        head = p / "score_head.pt"
        if head.exists(): state.update({f"score_head.{k}": v for k, v in torch.load(head, map_location="cpu").items()})
        return state
    ck = torch.load(p, map_location="cpu"); return ck.get("state_dict", ck)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v4", required=True); ap.add_argument("--v5", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--atol", type=float, default=0.0)
    a = ap.parse_args(); s4, s5 = load_state(a.v4), load_state(a.v5)
    shared = sorted(set(s4) & set(s5)); only4 = sorted(set(s4) - set(s5)); only5 = sorted(set(s5) - set(s4))
    changed, unchanged, rel = [], [], {}
    for k in shared:
        d = (s5[k].float() - s4[k].float()); n = float(s4[k].float().norm()) or 1.0
        rel[k] = float(d.norm()) / n
        (changed if d.abs().max().item() > a.atol else unchanged).append(k)
    lora = [k for k in shared if "lora_" in k]; head = [k for k in shared if k.startswith("score_head")]
    lora_changed = [k for k in changed if "lora_" in k]; head_changed = [k for k in changed if k.startswith("score_head")]
    verdict = ("YES: PPO changed LoRA/score-head tensors that produce s_dq (%d of %d LoRA tensors changed)" % (len(lora_changed), len(lora))
               if (lora_changed or head_changed) else
               "NO: all shared LoRA and score-head tensors are identical between V4 and V5; PPO did not touch the scoring weights")
    rep = {"verdict": verdict, "shared_tensors": len(shared), "changed": len(changed), "unchanged": len(unchanged),
           "lora_tensors": len(lora), "lora_changed": len(lora_changed), "score_head_tensors": len(head), "score_head_changed": len(head_changed),
           "max_relative_change": max(rel.values()) if rel else None, "median_relative_change": sorted(rel.values())[len(rel) // 2] if rel else None,
           "only_in_V4": only4[:20], "only_in_V5": only5[:20], "n_only_in_V5": len(only5)}
    print(json.dumps(rep, indent=1))
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
