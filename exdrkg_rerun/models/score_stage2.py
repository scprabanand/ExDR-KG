"""Step 10 / Step 2: score every held-out test pair with a Stage-2 model (base LLM + LoRA adapter + score head from stage2_finetune.py).

score = sigmoid(w . h_last(<SCORE>) + b): the same differentiable score head that was trained, read at the <SCORE> token that ends the prompt.
The <SCORE> embedding is rebuilt deterministically (add_score_token), exactly as in training. Prompt = retrieved paths + drug/disease, the same for
every backbone; no sampling, so the scores are reproducible. Missing pairs are an error, not silently dropped.

Input  --data data/llm/pairs.jsonl  (rows with split == "test"; must contain drug_id, disease_id, drug, disease, label, paths)
       --adapter models/stage2_<tag>_V4   (folder written by stage2_finetune.py: adapter, tokenizer, score_head.pt, run.json)
Output results/rerun/main/scores/<name>.csv   (drug_id, disease_id, label, score)  -> read by main_evaluation.py
Usage  python -m exdrkg_rerun.models.score_stage2 --adapter models/stage2_meta-llama_Llama-3.1-8B-Instruct_V4 --name "ExDR-KG-Llama3.1-8B (V4)"
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from exdrkg_rerun.models.stage2_finetune import build_prompt, add_score_token, SCORE_TOKEN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--data", default="data/llm/pairs.jsonl"); ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="."); ap.add_argument("--out", default="results/rerun/main/scores"); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--split", default="test"); a = ap.parse_args(); root = Path(a.root); ad = root / a.adapter
    run = json.load(open(ad / "run.json")); rows = [j for j in map(json.loads, open(root / a.data)) if j["split"] == a.split]
    need = {"drug_id", "disease_id", "drug", "disease", "label", "paths"}
    if not rows or not need <= set(rows[0]): raise SystemExit(f"{a.data} must have {sorted(need)} for split '{a.split}' (found {sorted(rows[0]) if rows else 'no rows'})")
    tok = AutoTokenizer.from_pretrained(ad); assert SCORE_TOKEN in tok.get_vocab(), "adapter tokenizer lacks <SCORE>"
    tok.padding_side = "right"; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(run["model"], revision=run.get("commit_hash") or run.get("revision_requested"), torch_dtype=torch.bfloat16, device_map="auto")
    add_score_token(lm, tok); lm = PeftModel.from_pretrained(lm, ad).eval()
    head = nn.Linear(lm.config.hidden_size, 1).to(lm.device, dtype=torch.float32); head.load_state_dict(torch.load(ad / "score_head.pt", map_location=lm.device)); head.eval()
    sid = tok.convert_tokens_to_ids(SCORE_TOKEN); scores = []
    for i in range(0, len(rows), a.batch_size):
        b = rows[i:i + a.batch_size]; ids = [(tok(build_prompt(r), add_special_tokens=True).input_ids + [sid])[-a.max_len:] for r in b]
        L = max(map(len, ids)); x = torch.tensor([t + [pad] * (L - len(t)) for t in ids], device=lm.device); m = torch.tensor([[1] * len(t) + [0] * (L - len(t)) for t in ids], device=lm.device)
        with torch.no_grad():
            h = lm(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states[-1]
            last = torch.tensor([len(t) - 1 for t in ids], device=lm.device); s = torch.sigmoid(head(h[torch.arange(len(b)), last].float())).squeeze(-1)
        scores += s.cpu().tolist()
        if (i // a.batch_size) % 100 == 0: print(f"{i}/{len(rows)}", flush=True)
    df = pd.DataFrame({"drug_id": [r["drug_id"] for r in rows], "disease_id": [r["disease_id"] for r in rows], "label": [r["label"] for r in rows], "score": scores})
    assert len(df) == len(rows) and df.score.notna().all()
    out = root / a.out; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / f"{a.name}.csv", index=False); print("wrote", out / f"{a.name}.csv", len(df))


if __name__ == "__main__":
    main()
