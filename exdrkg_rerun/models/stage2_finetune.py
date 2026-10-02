"""Step 10: Stage-2 LoRA fine-tuning with a DIFFERENTIABLE score and path-faithfulness loss, backbone-swappable.

Why this file exists
--------------------
In src/models/llm_reasoner.py the score is parsed from generated text and wrapped in torch.tensor(...), so the
prediction loss has no gradient path into the LoRA weights (compute_loss builds tensors from Python floats). That
means the repository cannot train the reasoner as described. This script fixes it:

  * prompt = retrieved typed evidence paths + drug/disease context, ending with a special <SCORE> token;
  * s_dq = sigmoid(w . h_last(<SCORE>) + b)  -> BCE with the label (gradient flows into LoRA and the head);
  * explanation tokens are trained with teacher forcing on a template built ONLY from the retrieved paths
    (language-model loss), and the faithfulness loss penalises probability mass on entity tokens that are not
    in the retrieved paths:  L_faith = mean_t -log(1 - sum_{v in F_t} p_t(v)),  F_t = first sub-word tokens of
    KG entity names absent from the evidence set of that example.
  * L = L_bce + lambda_lm * L_lm + lambda_faith * L_faith        (lambda_faith = 0.5, as in the manuscript)

Same settings for every backbone: LoRA rank 16, alpha 32, 10 epochs, lr 2e-5, seed 42, same encoder paths.
Backbones (record exact versions): BioMistral/BioMistral-7B, meta-llama/Llama-3.1-8B-Instruct or
Qwen/Qwen2.5-7B-Instruct, and a second biomedical model (e.g. epfl-llm/meditron-7b, aaditya/Llama3-OpenBioLLM-8B).

Input JSONL (one line per pair, built with the path retriever on the MASKED HKG):
  {"drug":..., "disease":..., "label":0/1, "paths":["A -[binds]-> B ...", ...], "entities":["A","B",...],
   "template":"explanation text using only path entities", "split":"train|val|test"}
Usage:
  python -m exdrkg_rerun.models.stage2_finetune --model BioMistral/BioMistral-7B --data data/llm/pairs.jsonl --out models/stage2_biomistral
  python -m exdrkg_rerun.models.stage2_finetune --model meta-llama/Llama-3.1-8B-Instruct --data ... --out models/stage2_llama --no-faith  (V3)
"""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

SCORE_TOKEN = "<SCORE>"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")


def add_score_token(lm, tok):
    """Add <SCORE> and give its embedding a DETERMINISTIC value (mean of the existing input embeddings), so that scoring later
    (score_stage2.py) rebuilds exactly the same token without needing the embedding matrix to be saved. Same for every backbone."""
    n0 = lm.get_input_embeddings().weight.shape[0]; lm.resize_token_embeddings(len(tok))
    with torch.no_grad():
        E = lm.get_input_embeddings().weight; E[n0:] = E[:n0].float().mean(0).to(E.dtype)
        if lm.get_output_embeddings() is not None and lm.get_output_embeddings().weight.data_ptr() != E.data_ptr():
            O = lm.get_output_embeddings().weight; O[n0:] = O[:n0].float().mean(0).to(O.dtype)


def build_prompt(r):
    paths = "\n".join(f"- {p}" for p in r["paths"][:10])
    return (f"Evidence paths from the knowledge graph:\n{paths}\n\nDrug: {r['drug']}\nDisease: {r['disease']}\n"
            f"Question: does the drug treat the disease? Explain using only the evidence above.\nExplanation:")


def forbidden_first_tokens(tok, all_entities, allowed):
    ids = set()
    for e in all_entities:
        if e in allowed: continue
        t = tok(" " + e, add_special_tokens=False).input_ids
        if t: ids.add(t[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda-faith", type=float, default=0.5); ap.add_argument("--lambda-lm", type=float, default=0.3)
    ap.add_argument("--no-faith", action="store_true", help="V3 setting: no faithfulness loss")
    ap.add_argument("--entity-vocab", default=None, help="JSON list of all KG entity names (defaults to union in data)")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--revision", default=None, help="exact HF revision (commit hash) of the backbone; recorded in run.json")
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]; train = [r for r in rows if r["split"] == "train"]; val = [r for r in rows if r["split"] == "val"]
    all_ents = json.load(open(a.entity_vocab)) if a.entity_vocab else sorted({e for r in rows for e in r["entities"]})
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision); tok.add_special_tokens({"additional_special_tokens": [SCORE_TOKEN]})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16, device_map="auto"); add_score_token(lm, tok)
    lm = get_peft_model(lm, LoraConfig(**LORA))
    commit = getattr(lm.base_model.model.config, "_commit_hash", None); n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad); n_all = sum(p.numel() for p in lm.parameters())
    torch.cuda.reset_peak_memory_stats(); hist = []
    head = nn.Linear(lm.config.hidden_size, 1).to(lm.device, dtype=torch.float32)
    score_id = tok.convert_tokens_to_ids(SCORE_TOKEN)
    params = [p for p in lm.parameters() if p.requires_grad] + list(head.parameters()); opt = torch.optim.AdamW(params, lr=a.lr)
    t0 = time.time()
    for ep in range(a.epochs):
        lm.train(); random.shuffle(train); tot = 0.0
        for r in train:
            prompt = build_prompt(r); expl = " " + r["template"]
            ids_p = tok(prompt, add_special_tokens=True).input_ids + [score_id]; ids_e = tok(expl, add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = torch.tensor([(ids_p + ids_e)[: a.max_len]], device=lm.device)
            out = lm(input_ids=ids, output_hidden_states=True)
            h_score = out.hidden_states[-1][0, len(ids_p) - 1].float()
            s = torch.sigmoid(head(h_score)).squeeze()
            l_bce = F.binary_cross_entropy(s, torch.tensor(float(r["label"]), device=s.device))
            logits = out.logits[0, len(ids_p) - 1: -1].float()                  # positions predicting explanation tokens
            tgt = ids[0, len(ids_p):]; n = min(len(tgt), logits.size(0)); logits, tgt = logits[:n], tgt[:n]
            l_lm = F.cross_entropy(logits, tgt)
            loss = l_bce + a.lambda_lm * l_lm
            if not a.no_faith:
                forb = forbidden_first_tokens(tok, all_ents, set(r["entities"]))
                if forb:
                    mass = F.softmax(logits, -1)[:, forb].sum(-1).clamp(max=1 - 1e-6)
                    loss = loss + a.lambda_faith * (-torch.log1p(-mass)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += loss.item()
        vl = None
        if val:
            lm.eval()
            with torch.no_grad():
                vs = []
                for r in val[:500]:
                    ip = tok(build_prompt(r), add_special_tokens=True).input_ids + [score_id]; o = lm(input_ids=torch.tensor([ip[: a.max_len]], device=lm.device), output_hidden_states=True)
                    vs.append(F.binary_cross_entropy(torch.sigmoid(head(o.hidden_states[-1][0, len(ip) - 1].float())).squeeze(), torch.tensor(float(r["label"]), device=lm.device)).item())
            vl = float(np.mean(vs))
        hist.append({"epoch": ep + 1, "train_loss": tot / max(len(train), 1), "val_bce_first500": vl})
        print(f"epoch {ep + 1}/{a.epochs} train loss {tot / max(len(train), 1):.4f} val bce {vl}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True); lm.save_pretrained(a.out); tok.save_pretrained(a.out)
    torch.save(head.state_dict(), Path(a.out) / "score_head.pt")
    json.dump({"model": a.model, "revision_requested": a.revision, "commit_hash": commit, "stage2_hours": (time.time() - t0) / 3600, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
               "faithfulness": not a.no_faith, "lambda_faith": 0.0 if a.no_faith else a.lambda_faith, "lambda_lm": a.lambda_lm, "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "lora": LORA,
               "trainable_params": n_train, "total_params": n_all, "n_train": len(train), "n_val": len(val), "data": a.data, "data_md5": hashlib.md5(open(a.data, "rb").read()).hexdigest(),
               "history": hist, "final_epoch_model_saved": True, "rlhf": False}, open(Path(a.out) / "run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
