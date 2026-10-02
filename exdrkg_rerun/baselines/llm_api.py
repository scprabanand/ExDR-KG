"""Step 11 (Comments 2.6, 2.7): updated LLM baselines through any OpenAI-compatible chat API, temperature 0.

Two contexts, one identical prompt template:
  --context rag    recent model + text RAG. The passages are the SAME ones as in the original GPT-4+RAG pipeline (same corpus, chunking,
                   embedder, index, top-k, reranker): they are read from the input file ("passages"), never re-retrieved here.
  --context paths  the same recent model + the top-10 ExDR-KG evidence paths given to the reasoning engine, ZERO-SHOT (no fine-tuning).
The model returns JSON {"probability": p, "explanation": text}. The prompt template is fixed in this file (its md5 is stored in the run record);
to reuse the ORIGINAL GPT-4+RAG prompt word for word, put it in a text file with the placeholders {context_name} {context} {drug} {disease}
and pass --prompt-file (the file is then used for both contexts, and its md5 is recorded).

Reproducibility and cost control
  * temperature 0, seed 42 (best effort at the API), max_tokens 400; the served model id (resp.model), system_fingerprint and token usage are
    recorded per call; query dates (first/last) and the exact model string go to run_<tag>.json for Table R-K.
  * FULL set = every test pair. If cost is a limit use --sample-negatives N: all positives + a fixed random sample of N negatives (seed 42).
    The sample is written once to <out>/sample_ids.csv and RE-USED (--sample-file) by every later model, so all models see the same pairs.
  * API failures are NOT scored as 0.5: they are marked api_error, retried on resume, and counted. Unparsable answers are counted (parsed=False);
    probabilities in "85%" form are converted; values are clipped to [0, 1]. Both counts are printed and stored; for the metrics an unanswered
    pair gets score 0.5 (a tie) and the summary reports how many.
  * concurrency: --workers threads (default 4); output order = input order.
Input JSONL per pair: {"drug_id","disease_id","drug","disease","label","passages":[...],"paths":[...]}  (paths built with build_evidence.py on the MASKED HKG)
Usage
  export OPENAI_API_KEY=...            # never stored in any file
  python -m exdrkg_rerun.baselines.llm_api --model gpt-4o-2024-08-06 --context rag   --data data/llm/test_pairs.jsonl --out results/rerun/llm --score-name "GPT-4o+RAG"
  python -m exdrkg_rerun.baselines.llm_api --model gpt-4o-2024-08-06 --context paths --data data/llm/test_pairs.jsonl --out results/rerun/llm --score-name "GPT-4o+ExDR-KG paths (zero-shot)"
  add --sample-negatives 5000 --sample-file results/rerun/llm/sample_ids.csv if the budget requires a sample.
"""
import argparse, csv, datetime, hashlib, json, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI

SYSTEM = "You are a biomedical expert. Answer with JSON only."
TEMPLATE = ("{context_name}:\n{context}\n\nDrug: {drug}\nDisease: {disease}\n"
            'Question: how likely is it that this drug is an effective treatment for this disease? '
            'Answer as JSON: {{"probability": <number between 0 and 1>, "explanation": "<mechanistic explanation>"}}. '
            "Use only entities that appear in the context above.")


def parse_prob(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): x = float(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+(\.\d+)?", v)
        if not m: return None
        x = float(m.group(0)); x = x / 100 if ("%" in v or x > 1) and x <= 100 else x
    else: return None
    return min(1.0, max(0.0, x))


def call(client, model, msg, seed):
    err = None
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=model, temperature=0, seed=seed, max_tokens=400, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}])
            u = r.usage
            return r.choices[0].message.content or "", {"served_model": r.model, "system_fingerprint": getattr(r, "system_fingerprint", None),
                                                          "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None)}, None
        except Exception as e:
            err = repr(e)[:300]; time.sleep(min(60, 2 ** attempt))
    return "", {}, err


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--context", choices=["rag", "paths"], required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--base-url", default=None)
    ap.add_argument("--score-name", default=None, help="also write results/rerun/main/scores/<name>.csv (full set only; a sample is written to the sample_scores folder)")
    ap.add_argument("--sample-negatives", type=int, default=None); ap.add_argument("--sample-file", default=None); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-file", default=None); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=None, help="smoke test: first N rows only")
    a = ap.parse_args(); client = OpenAI(base_url=a.base_url) if a.base_url else OpenAI()
    tmpl = open(a.prompt_file).read() if a.prompt_file else TEMPLATE; prompt_md5 = hashlib.md5((SYSTEM + tmpl).encode()).hexdigest()
    rows = [json.loads(l) for l in open(a.data)]; key = lambda r: f"{r['drug_id']}|{r['disease_id']}"
    field = "passages" if a.context == "rag" else "paths"
    if any(field not in r for r in rows[:50]): raise SystemExit(f"input rows have no '{field}' field; build it first (RAG passages: original corpus/retriever; paths: build_evidence.py)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sf = Path(a.sample_file) if a.sample_file else out / "sample_ids.csv"
    if a.sample_negatives is not None:
        if sf.exists(): keep = {l.strip() for l in open(sf) if l.strip() and l.strip() != "pair"}; print(f"re-using sample {sf} ({len(keep)} pairs)")
        else:
            rnd = random.Random(a.seed); pos = [r for r in rows if r["label"] == 1]; neg = sorted([r for r in rows if r["label"] == 0], key=key)
            keep = {key(r) for r in pos + rnd.sample(neg, min(a.sample_negatives, len(neg)))}; sf.write_text("pair\n" + "\n".join(sorted(keep)) + "\n"); print(f"wrote sample {sf} ({len(keep)} pairs)")
        rows = [r for r in rows if key(r) in keep]
    if a.limit: rows = rows[:a.limit]
    tag = f"{a.model.replace('/', '_')}_{a.context}"; fpath = out / f"expl_{tag}.jsonl"; done = {}
    if fpath.exists():                                         # resume: keep good rows, retry api errors
        for l in open(fpath):
            j = json.loads(l)
            if not j.get("api_error"): done[f"{j['drug_id']}|{j['disease_id']}"] = j
    todo = [r for r in rows if key(r) not in done]; lock = threading.Lock(); fx = open(fpath, "w")
    for j in done.values(): fx.write(json.dumps(j) + "\n")
    first_ts = last_ts = None
    def work(r):
        ctx_items = r[field][:10]; ctx = "\n".join(f"- {p}" for p in ctx_items)
        msg = tmpl.format(context_name="Retrieved passages" if a.context == "rag" else "Evidence paths from a knowledge graph", context=ctx, drug=r["drug"], disease=r["disease"])
        txt, meta, err = call(client, a.model, msg, a.seed); m = re.search(r"\{.*\}", txt, re.S)
        try: j = json.loads(m.group(0)) if m else {}
        except Exception: j = {}
        p = parse_prob(j.get("probability")); rec = {"drug_id": r["drug_id"], "disease_id": r["disease_id"], "label": r["label"], "score": 0.5 if p is None else p, "parsed": p is not None,
                                                     "n_context_items": len(ctx_items), "explanation": j.get("explanation", txt), "api_error": err, "timestamp": datetime.datetime.utcnow().isoformat(), **meta}
        with lock: fx.write(json.dumps(rec) + "\n"); fx.flush(); done[key(r)] = rec
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, todo)):
            if i % 200 == 0: print(f"{i}/{len(todo)}", flush=True)
    fx.close(); recs = [done[key(r)] for r in rows if key(r) in done]
    ts = sorted(x["timestamp"] for x in recs if x.get("timestamp")); n_err = sum(bool(x.get("api_error")) for x in recs)
    def write(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["drug_id", "disease_id", "label", "score"])
            for x in recs: w.writerow([x["drug_id"], x["disease_id"], x["label"], x["score"]])
    write(out / f"scores_{tag}.csv")
    if a.score_name: write(Path("results/rerun/main/scores" if a.sample_negatives is None else "results/rerun/main/sample_scores") / f"{a.score_name}.csv")
    served = sorted({x.get("served_model") for x in recs if x.get("served_model")}); fp = sorted({x.get("system_fingerprint") for x in recs if x.get("system_fingerprint")})
    info = {"model_requested": a.model, "served_model_ids": served, "system_fingerprints": fp, "context": a.context, "temperature": 0, "seed": a.seed, "query_first_utc": ts[0] if ts else None,
            "query_last_utc": ts[-1] if ts else None, "n_pairs": len(recs), "n_positives": sum(x["label"] == 1 for x in recs), "api_errors": n_err, "unparsed": sum(not x["parsed"] for x in recs),
            "prompt_md5": prompt_md5, "prompt_file": a.prompt_file, "context_items_mean": sum(x["n_context_items"] for x in recs) / max(1, len(recs)),
            "pairs_with_empty_context": sum(x["n_context_items"] == 0 for x in recs), "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in recs),
            "completion_tokens": sum(x.get("completion_tokens") or 0 for x in recs), "sample_negatives": a.sample_negatives, "sample_file": str(sf) if a.sample_negatives is not None else None, "data": a.data}
    json.dump(info, open(out / f"run_{tag}.json", "w"), indent=1); print(json.dumps(info, indent=1))
    if n_err: print(f"WARNING: {n_err} pairs still have API errors; rerun the same command to retry them before using the scores")


if __name__ == "__main__":
    main()
