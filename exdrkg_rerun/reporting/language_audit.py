"""Step 14 (Comments 1.6, 1.7): language audit of the manuscript (.docx). Reports problems, changes nothing unless you accept a fix.

Subcommands
  audit     read the manuscript and write results/rerun/language/ :
              missing_spaces.csv   glued or split words: "bi-ological", "Aand Bare", "0recovers", "methodThe", no space after . , ; :, "the the" repeats,
                                   words that are not in the dictionary but split into two dictionary words ("cellsare"), hyphen-split fragments whose
                                   joined form is a word ("bi-ological" -> "biological");
              duplicate_sentences.csv  exact and near-duplicate sentences (normalised; near = difflib ratio >= 0.90) with paragraph numbers;
              acronyms.csv         for every acronym: first use, definition(s), number of uses, and a status such as USED_BEFORE_DEFINED, DEFINED_TWICE,
                                   EXPANDED_AGAIN_AFTER_DEFINITION, NEVER_DEFINED, DEFINED_NEVER_USED. LLM, KG, HKG, RLHF, PPO, LoRA and MMR are always checked;
              heuristics.csv       double spaces, space before punctuation, unbalanced brackets/quotes, "a/an" mistakes, mixed British/American spelling
                                   pairs (the manuscript is British: generalisation, ...), sentences over 60 words;
              audit_summary.md     counts and the first 30 hits of each kind.
  grammar   full grammar pass. Either --engine languagetool (language_tool_python, needs Java and a download; en-GB) or --engine llm (any OpenAI-compatible
              endpoint, temperature 0: the model must return JSON edits {"original","corrected","reason"} for each paragraph, is forbidden to change numbers,
              citations, symbols, acronyms or meaning, and every edit is checked to be a substring of the paragraph). Writes grammar_suggestions.csv with an empty
              `accept` column: you review it (1 = accept).
  apply     write <name>_language_fixed.docx (the original is never overwritten) applying rows with accept = 1 from grammar_suggestions.csv and the
              --auto-fix rules that are unambiguous (double spaces, space before , . ; :, space after a full stop when the next word is capitalised and the text
              before is not a known abbreviation). Edits are applied inside a single run to keep formatting; edits that span runs are listed in apply_report.csv.
Usage
  python -m exdrkg_rerun.reporting.language_audit audit   --docx "Manuscript_Revised_V1.docx"
  python -m exdrkg_rerun.reporting.language_audit grammar --docx "Manuscript_Revised_V1.docx" --engine llm --model gpt-4o-2024-08-06     (needs OPENAI_API_KEY)
  python -m exdrkg_rerun.reporting.language_audit apply   --docx "Manuscript_Revised_V1.docx" --auto-fix
Needs: python-docx, pyspellchecker (dictionary), optionally language_tool_python or openai.
"""
import argparse, difflib, json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from docx import Document

SEVEN = {"LLM": r"large language models?", "KG": r"knowledge graphs?", "HKG": r"heterogeneous knowledge graphs?", "RLHF": r"reinforcement learning from human feedback",
         "PPO": r"proximal policy optimi[sz]ation", "LoRA": r"low[- ]rank adaptation", "MMR": r"maxim(al|um) marginal relevance"}
ABBR_OK = {"e.g", "i.e", "et al", "vs", "Fig", "Figs", "Eq", "Eqs", "Dr", "No", "approx", "cf", "Ref", "Refs", "Sec", "Tab"}
UK_US = [("generalisation", "generalization"), ("optimisation", "optimization"), ("normalisation", "normalization"), ("visualisation", "visualization"), ("colour", "color"),
         ("behaviour", "behavior"), ("centre", "center"), ("analyse", "analyze"), ("modelling", "modeling"), ("labelled", "labeled"), ("regularisation", "regularization"), ("utilise", "utilize")]
ACR_STOP = {"AND", "THE", "FOR", "WITH", "NOT", "USA", "UK", "II", "III", "IV", "PDF", "DOI", "URL", "ID"}


def paragraphs(doc):
    out = []
    for p in doc.paragraphs:
        if p.text.strip(): out.append(p.text)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    if p.text.strip(): out.append(p.text)
    return out


def speller():
    try:
        from spellchecker import SpellChecker
        return SpellChecker(language="en")
    except Exception: print("pyspellchecker not installed: dictionary-based checks are skipped (pip install pyspellchecker)"); return None


def missing_spaces(paras, sp):
    hits = []
    known = set()
    for i, t in enumerate(paras, 1):
        for m in re.finditer(r"\b([a-z]{1,4})-([a-z]{3,})\b", t):
            j = m.group(1) + m.group(2)
            if sp and j in sp and m.group(1) not in sp: hits.append((i, m.group(0), f"hyphen-split word -> {j}"))
        for m in re.finditer(r"\b[a-z]+[A-Z][a-z]+\b", t):
            if m.group(0) not in ("LoRA", "BioGPT", "BioBERT", "PyTorch", "DrugBank", "DisGeNET", "TxGNN", "ChatGPT", "GitHub", "PubMed", "MetaPath"): hits.append((i, m.group(0), "lower-case letter followed by capital (glued words?)"))
        for m in re.finditer(r"\b[A-Z][a-z]{0,2}(and|are|is|of|the|to|in|for)\s+[A-Z]?[a-z]*\b", t):
            if len(m.group(0)) < 14 and m.group(0).split()[0] not in ("Sand", "Band", "Hand", "Land", "Brand", "Grand", "Stand", "Island", "Thanks", "Therefore", "Theory"): hits.append((i, m.group(0), "possible glued 'Aand Bare' pattern"))
        for m in re.finditer(r"[=<>≥≤±×μσλαβ]\s*\d+(\.\d+)?[A-Za-z]{3,}", t): hits.append((i, m.group(0), "number glued to following word"))
        for m in re.finditer(r"(?<![\w.])\d+(\.\d+)?[a-z]{4,}\b", t):
            if not re.match(r"\d+(st|nd|rd|th|fold|hop|dim|layer|epoch|bit)", m.group(0)): hits.append((i, m.group(0), "digit glued to a word"))
        for m in re.finditer(r"[a-z]{2,}[,;:][A-Za-z]", t): hits.append((i, m.group(0), "no space after punctuation"))
        for m in re.finditer(r"[a-z]{3,}\.[A-Z][a-z]{2,}", t):
            if not re.search(r"(www|http|\.com|\.org|\.py|\.pt|\.csv|e\.g|i\.e)", t[max(0, m.start() - 6): m.end() + 6]): hits.append((i, m.group(0), "no space after full stop"))
        for m in re.finditer(r"\b(\w+)\s+\1\b", t, re.I):
            if m.group(1).lower() not in ("had", "that") and not m.group(1).isdigit(): hits.append((i, m.group(0), "repeated word"))
        if sp:
            for w in re.findall(r"\b[a-z]{7,}\b", t):
                if w in known or w in sp: continue
                known.add(w)
                for k in range(3, len(w) - 2):
                    a_, b_ = w[:k], w[k:]
                    if len(a_) >= 3 and len(b_) >= 3 and a_ in sp and b_ in sp and w not in sp and (a_ not in ("the", "and") or True):
                        hits.append((i, w, f"not a word; split as '{a_} {b_}'")); break
    return pd.DataFrame(hits, columns=["paragraph", "text", "issue"]).drop_duplicates()


def sentences(paras):
    S = []
    for i, t in enumerate(paras, 1):
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", t):
            n = re.sub(r"[^a-z0-9 ]", "", s.lower()); n = re.sub(r"\s+", " ", n).strip()
            if len(n.split()) >= 6: S.append((i, s.strip(), n))
    return S


def duplicates(paras):
    S = sentences(paras); rows = []; seen = defaultdict(list)
    for i, s, n in S: seen[n].append((i, s))
    for n, v in seen.items():
        if len(v) > 1: rows.append({"kind": "exact", "paragraphs": ",".join(str(x[0]) for x in v), "ratio": 1.0, "sentence": v[0][1][:300], "other": ""})
    uniq = [(i, s, n) for n, v in seen.items() for (i, s) in v[:1] for _ in [0] for n in [n]]
    for x in range(len(uniq)):
        for y in range(x + 1, len(uniq)):
            a_, b_ = uniq[x], uniq[y]
            if abs(len(a_[2]) - len(b_[2])) > 0.2 * max(len(a_[2]), len(b_[2])): continue
            r = difflib.SequenceMatcher(None, a_[2], b_[2]).ratio()
            if r >= 0.90: rows.append({"kind": "near", "paragraphs": f"{a_[0]},{b_[0]}", "ratio": round(r, 3), "sentence": a_[1][:300], "other": b_[1][:300]})
    return pd.DataFrame(rows)


def acronyms(paras):
    text = "\n".join(paras); defs = defaultdict(list)
    for m in re.finditer(r"([A-Za-z][A-Za-z\- ]{3,80}?)\s*\(([A-Z][A-Za-z0-9\-]{1,9}s?)\)", text):
        long, ac = m.group(1).strip(), m.group(2)
        if ac.isupper() or ac in SEVEN or re.search(r"[A-Z]{2,}", ac):
            words = re.findall(r"[A-Za-z]+", long)[-len(re.sub(r"[^A-Z]", "", ac)) - 1:]
            defs[ac.rstrip("s") if ac.endswith("s") and ac[:-1].isupper() else ac].append((m.start(), long))
    cand = Counter(m.group(0) for m in re.finditer(r"\b[A-Z][A-Za-z]?[A-Z]{1,7}[a-z]?\b", text)); rows = []
    for ac in sorted(set(defs) | {k for k, v in cand.items() if v >= 3 and k not in ACR_STOP and len(k) >= 2} | set(SEVEN)):
        uses = [m.start() for m in re.finditer(rf"(?<![A-Za-z]){re.escape(ac)}s?(?![A-Za-z])", text)]; d = defs.get(ac, []); dpos = sorted(p for p, _ in d)
        # a definition occurrence includes the acronym itself in brackets, so uses beyond definitions = len(uses) - len(d)
        first_use = uses[0] if uses else None; status = []
        if not d: status.append("NEVER_DEFINED" if uses else "ABSENT")
        if len(d) > 1: status.append("DEFINED_TWICE")
        if d and first_use is not None and first_use < dpos[0] - len(ac) - 2 and first_use < min(dpos): status.append("USED_BEFORE_DEFINED")
        if d and len(uses) - len(d) <= 0: status.append("DEFINED_NEVER_USED")
        if ac in SEVEN:
            exp = [m.start() for m in re.finditer(SEVEN[ac], text, re.I)]; after = [p for p in exp if dpos and p > dpos[0] + 5]
            n_after = max(0, len(after) - (len(d) - 1))
            if n_after > 0: status.append(f"EXPANDED_AGAIN_AFTER_DEFINITION({n_after}x)")
        rows.append({"acronym": ac, "definitions": len(d), "definition_text": " | ".join(x[1] for x in d)[:150], "uses_incl_definitions": len(uses), "first_use_char": first_use, "status": ";".join(status) or "ok", "priority": ac in SEVEN})
    return pd.DataFrame(rows).sort_values(["priority", "status"], ascending=[False, True])


def heuristics(paras):
    rows = []; uk = {u: 0 for u, _ in UK_US}; us = {v: 0 for _, v in UK_US}; full = " ".join(paras).lower()
    for i, t in enumerate(paras, 1):
        for pat, msg in ((r"  +", "double space"), (r"\s[,.;:](?!\d)", "space before punctuation"), (r"\ba ([aeiou]\w+)", "'a' before vowel sound"), (r"\ban ([bcdfgjklmnpqrstvwxz]\w+)", "'an' before consonant")):
            for m in re.finditer(pat, t):
                if msg.startswith("'a'") and m.group(1).lower().startswith(("uni", "use", "eu", "one", "ubiq")): continue
                if msg.startswith("'an'") and m.group(1).lower().startswith(("hour", "honest", "mri", "hmm")): continue
                rows.append((i, m.group(0), msg))
        if t.count("(") != t.count(")") or t.count("[") != t.count("]") or t.count('"') % 2: rows.append((i, t[:60], "unbalanced brackets or quotes"))
        for s in re.split(r"(?<=[.!?])\s+", t):
            if len(s.split()) > 60: rows.append((i, s[:80], f"very long sentence ({len(s.split())} words)"))
    for u, v in UK_US:
        if full.count(u) and full.count(v): rows.append((0, f"{u}/{v}", f"mixed spelling: {full.count(u)} British vs {full.count(v)} American"))
    return pd.DataFrame(rows, columns=["paragraph", "text", "issue"])


def audit(a):
    doc = Document(a.docx); paras = paragraphs(doc); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sp = speller()
    M = missing_spaces(paras, sp); D = duplicates(paras); A = acronyms(paras); H = heuristics(paras)
    M.to_csv(out / "missing_spaces.csv", index=False); D.to_csv(out / "duplicate_sentences.csv", index=False); A.to_csv(out / "acronyms.csv", index=False); H.to_csv(out / "heuristics.csv", index=False)
    md = [f"# Language audit of {a.docx}", f"paragraphs: {len(paras)}", f"- glued/split words and spacing: {len(M)}", f"- duplicate sentences: {len(D)} ({int((D.kind == 'exact').sum()) if len(D) else 0} exact)",
          f"- acronyms with a problem: {int((A.status != 'ok').sum())} of {len(A)}", f"- other heuristic issues: {len(H)}", "", "## Acronyms to fix (the seven required first)", A[A.status != "ok"].head(40).to_markdown(index=False), "",
          "## Missing spaces (first 30)", M.head(30).to_markdown(index=False), "", "## Duplicates (first 30)", D.head(30).to_markdown(index=False) if len(D) else "none", "", "## Heuristics (first 30)", H.head(30).to_markdown(index=False)]
    (out / "audit_summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:8]))


def grammar(a):
    doc = Document(a.docx); paras = paragraphs(doc); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    if a.engine == "languagetool":
        import language_tool_python; tool = language_tool_python.LanguageTool("en-GB")
        for i, t in enumerate(paras, 1):
            for m in tool.check(t):
                if m.replacements and m.ruleId not in ("MORFOLOGIK_RULE_EN_GB", "UPPERCASE_SENTENCE_START"):
                    rows.append({"paragraph": i, "original": t[m.offset:m.offset + m.errorLength], "corrected": m.replacements[0], "reason": f"{m.ruleId}: {m.message}", "context": t[max(0, m.offset - 40): m.offset + m.errorLength + 40], "accept": ""})
    else:
        from openai import OpenAI; cl = OpenAI(base_url=a.base_url) if a.base_url else OpenAI()
        sysm = ("You are a copy editor for a biomedical AI paper (British English). Find grammar, spelling, agreement, article, punctuation and missing-space errors and clumsy wording. "
                "NEVER change numbers, units, citations like [12], symbols, equations, acronyms, model names or scientific meaning. Return JSON: {\"edits\":[{\"original\":<exact substring>,\"corrected\":<replacement>,\"reason\":<short>}]}. "
                "Each original must be copied exactly from the paragraph and be as short as possible. Return {\"edits\":[]} if the paragraph is fine.")
        for i, t in enumerate(paras, 1):
            if len(t.split()) < 6: continue
            r = cl.chat.completions.create(model=a.model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": sysm}, {"role": "user", "content": t}])
            try: ed = json.loads(r.choices[0].message.content).get("edits", [])
            except Exception: ed = []
            for e in ed:
                if e.get("original") and e["original"] in t and e.get("corrected") and e["corrected"] != e["original"]:
                    if re.findall(r"\d+(?:\.\d+)?", e["original"]) != re.findall(r"\d+(?:\.\d+)?", e["corrected"]): continue      # never touch numbers
                    rows.append({"paragraph": i, "original": e["original"], "corrected": e["corrected"], "reason": e.get("reason", ""), "context": t[max(0, t.find(e["original"]) - 40): t.find(e["original"]) + len(e["original"]) + 40], "accept": ""})
    pd.DataFrame(rows).to_csv(out / "grammar_suggestions.csv", index=False); print(f"{len(rows)} suggestions -> {out / 'grammar_suggestions.csv'}; set accept=1 for the ones you agree with, then run 'apply'")


AUTO = [(re.compile(r"[ \t]{2,}"), " "), (re.compile(r"\s+([,;:.])(?=\s|$)"), r"\1"), (re.compile(r"\b([A-Za-z]{3,}[a-z])([.,;:])([A-Z][a-z]{2,})"), None)]


def apply(a):
    doc = Document(a.docx); out = Path(a.out); sug = pd.read_csv(out / "grammar_suggestions.csv") if (out / "grammar_suggestions.csv").exists() else pd.DataFrame(columns=["original", "corrected", "accept"])
    edits = [(str(r.original), str(r.corrected)) for r in sug.itertuples() if str(getattr(r, "accept", "")).strip() in ("1", "1.0")]; rep = []
    def all_paras():
        yield from doc.paragraphs
        for t in doc.tables:
            for row in t.rows:
                for c in row.cells: yield from c.paragraphs
    for p in all_paras():
        for o, n in edits:
            if o in p.text:
                done = False
                for r in p.runs:
                    if o in r.text: r.text = r.text.replace(o, n, 1); done = True; break
                rep.append({"original": o, "corrected": n, "applied": done, "note": "" if done else "spans several runs: edit by hand"})
        if a.auto_fix:
            for r in p.runs:
                t = re.sub(r"[ \t]{2,}", " ", r.text); t = re.sub(r" +([,;:])(?=\s|$)", r"\1", t); t = re.sub(r"\b([a-z]{4,})\.([A-Z][a-z]{2,})", lambda m: m.group(0) if m.group(1) in ABBR_OK else f"{m.group(1)}. {m.group(2)}", t)
                if t != r.text: rep.append({"original": r.text[:60], "corrected": t[:60], "applied": True, "note": "auto-fix"}); r.text = t
    dst = Path(a.docx).with_name(Path(a.docx).stem + "_language_fixed.docx"); doc.save(dst); pd.DataFrame(rep).to_csv(out / "apply_report.csv", index=False); print(f"wrote {dst}; {len(rep)} edits, report {out / 'apply_report.csv'}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("audit", "grammar", "apply"): p = sub.add_parser(n); p.add_argument("--docx", required=True); p.add_argument("--out", default="results/rerun/language")
    g = sub.choices["grammar"]; g.add_argument("--engine", choices=["languagetool", "llm"], default="llm"); g.add_argument("--model", default="gpt-4o-2024-08-06"); g.add_argument("--base-url", default=None)
    sub.choices["apply"].add_argument("--auto-fix", action="store_true"); a = ap.parse_args(); {"audit": audit, "grammar": grammar, "apply": apply}[a.cmd](a)


if __name__ == "__main__":
    main()
