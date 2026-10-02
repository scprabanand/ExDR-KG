"""Step 4, item 3: generate explanations for H1 and H2 with every model, into one common JSONL format.

Output line (all kinds): {"model","set","pair_id","explanation"}  (set is H1 or H2; a pair in both sets is generated once and written twice)
Decoding is deterministic everywhere (greedy / temperature 0, seed 42) so results can be reproduced.

Kinds
  biogpt         ungrounded BioGPT: prompt has only drug and disease names.
  lora           fine-tuned reasoner (V3, V4 or V5): base model + LoRA adapter from stage2_finetune.py, prompt = retrieved paths
                 (same builder as training). Run once per variant: --model-name V3 / V4 / V5 with the matching --adapter.
  api_input      write the input file for llm_api.py (GPT-4+RAG needs --passages, a JSONL of {"pair_id","passages":[...]}).
  api_convert    convert the explanations written by llm_api.py into the common format (--model-name GPT-4+RAG).
  txgnn_template template explanation filled from the top TxGNN explainer path (--txgnn-paths JSONL: {"pair_id","gene","pathway"}).
Usage examples
  python -m exdrkg_rerun.explainability.generate_explanations --kind biogpt --model-name BioGPT --hf-model microsoft/biogpt
  python -m exdrkg_rerun.explainability.generate_explanations --kind lora --model-name V4 --hf-model BioMistral/BioMistral-7B --adapter models/stage2_BioMistral_BioMistral-7B_V4
  python -m exdrkg_rerun.explainability.generate_explanations --kind api_input --passages data/llm/rag_passages.jsonl --out-api data/llm/hallu_pairs.jsonl
  python -m exdrkg_rerun.explainability.generate_explanations --kind api_convert --model-name GPT-4+RAG --api-expl results/rerun/llm/expl_gpt-4o-2024-08-06_rag.jsonl
"""
import argparse, json, sys
from pathlib import Path
import torch

BIOGPT_PROMPT = "Explain the molecular mechanism by which {drug} could be an effective treatment for {disease}."
TEMPLATE = ("{drug} may treat {disease} by acting on {gene}, a gene that participates in {pathway}, "
            "which is implicated in the pathophysiology of {disease}.")


def load_sets(sd: Path):
    sets = {}
    for s in ("H1", "H2"):
        sets[s] = [l.strip() for l in open(sd / f"{s}_pairs.txt") if l.strip()]
    return sets


def load_evidence(sd: Path):
    return {j["pair_id"]: j for j in map(json.loads, open(sd / "evidence.jsonl"))}


def emit(fout, model, sets, texts):
    for s, ids in sets.items():
        for p in ids:
            if p in texts: fout.write(json.dumps({"model": model, "set": s, "pair_id": p, "explanation": texts[p]}) + "\n")


def gen_hf(model, tok, prompts, max_new, dev, bs=8):
    out = []
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["biogpt", "lora", "api_input", "api_convert", "txgnn_template"], required=True)
    ap.add_argument("--model-name", default=None); ap.add_argument("--hf-model", default=None); ap.add_argument("--adapter", default=None)
    ap.add_argument("--sets", default="results/rerun/hallucination"); ap.add_argument("--root", default="."); ap.add_argument("--device", default="cuda")
    ap.add_argument("--passages", default=None); ap.add_argument("--out-api", default=None); ap.add_argument("--api-expl", default=None)
    ap.add_argument("--txgnn-paths", default=None); ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); torch.manual_seed(a.seed); sd = Path(a.root) / a.sets; sets = load_sets(sd); ev = load_evidence(sd)
    allp = sorted({p for v in sets.values() for p in v}); out_path = sd / "explanations.jsonl"
    if a.kind == "api_input":
        pas = {j["pair_id"]: j["passages"] for j in map(json.loads, open(a.passages))}
        with open(a.out_api, "w") as f:
            for p in allp:
                d, s = p.split("|"); f.write(json.dumps({"drug_id": d, "disease_id": s, "drug": ev[p]["drug"], "disease": ev[p]["disease"], "label": 0,
                                                          "passages": pas.get(p, []), "paths": ev[p]["paths"]}) + "\n")
        print("wrote", a.out_api); return
    texts = {}
    if a.kind == "api_convert":
        for j in map(json.loads, open(a.api_expl)): texts[f"{j['drug_id']}|{j['disease_id']}"] = j["explanation"]
    elif a.kind == "txgnn_template":
        for j in map(json.loads, open(a.txgnn_paths)):
            e = ev.get(j["pair_id"]);
            if e: texts[j["pair_id"]] = TEMPLATE.format(drug=e["drug"], disease=e["disease"], gene=j["gene"], pathway=j["pathway"])
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(a.hf_model)
        lm = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=torch.bfloat16 if a.kind == "lora" else torch.float16).to(a.device).eval()
        if a.kind == "biogpt":
            prompts = [BIOGPT_PROMPT.format(drug=ev[p]["drug"], disease=ev[p]["disease"]) for p in allp]
        else:
            from peft import PeftModel
            from exdrkg_rerun.models.stage2_finetune import build_prompt
            lm = PeftModel.from_pretrained(lm, a.adapter).eval()
            prompts = [build_prompt({"paths": ev[p]["paths"], "drug": ev[p]["drug"], "disease": ev[p]["disease"]}) for p in allp]
        for p, t in zip(allp, gen_hf(lm, tok, prompts, a.max_new_tokens, a.device)): texts[p] = t
    missing = [p for p in allp if p not in texts]
    if missing: print(f"WARNING: {len(missing)} pairs have no explanation for {a.model_name}; they are counted in N_missing by hallucination_eval.py")
    with open(out_path, "a") as f: emit(f, a.model_name, sets, texts)
    print(f"{a.model_name}: {len(texts)} explanations appended to {out_path}")


if __name__ == "__main__":
    main()
