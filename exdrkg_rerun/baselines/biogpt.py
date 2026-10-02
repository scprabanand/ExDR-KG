"""Step 2 (model 6): BioGPT baseline scored deterministically on every test pair.

BioGPT has no classification head, so the score is the model's own next-token preference between " yes" and " no"
after a fixed prompt: score = sigmoid(logP(" yes") - logP(" no")). This is reproducible (no sampling, no text parsing)
and gives a continuous score for AUROC/AUPRC/ranking. It replaces "ask the model for a calibrated probability" which
is not deterministic and needs parsing; state the choice in the manuscript.
The same prompt is used with only the drug and disease names (ungrounded baseline).

Input  data/llm/test_pairs.jsonl  ({"drug_id","disease_id","drug","disease","label"} per test pair, all 11,615 pairs)
Usage  python -m exdrkg_rerun.baselines.biogpt --data data/llm/test_pairs.jsonl --model microsoft/biogpt --out results/rerun/main/scores
       (use microsoft/BioGPT-Large for the larger variant; record the exact model name and revision)
"""
import argparse, json
from pathlib import Path
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "Question: Is {drug} an effective treatment for {disease}? Answer (yes or no):"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", default="microsoft/biogpt")
    ap.add_argument("--out", default="results/rerun/main/scores"); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); torch.manual_seed(42)
    rows = [json.loads(l) for l in open(a.data)]
    tok = AutoTokenizer.from_pretrained(a.model); lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(a.device).eval()
    yes = tok(" yes", add_special_tokens=False).input_ids[0]; no = tok(" no", add_special_tokens=False).input_ids[0]
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    scores = []
    for i in range(0, len(rows), a.batch_size):
        b = rows[i:i + a.batch_size]
        enc = tok([PROMPT.format(drug=r["drug"], disease=r["disease"]) for r in b], return_tensors="pt", padding=True).to(a.device)
        with torch.no_grad(): lp = torch.log_softmax(lm(**enc).logits[:, -1].float(), -1)
        scores += torch.sigmoid(lp[:, yes] - lp[:, no]).cpu().tolist()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"drug_id": [r["drug_id"] for r in rows], "disease_id": [r["disease_id"] for r in rows], "label": [r["label"] for r in rows], "score": scores}).to_csv(out / "BioGPT.csv", index=False)
    json.dump({"model": a.model, "prompt": PROMPT, "n": len(rows)}, open(out / "BioGPT.meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
