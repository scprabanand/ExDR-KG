"""Step 5, Table R-E3 (Comment 2.5): knowledge-graph harmonisation counts.

The rerun plan says these counts come "directly from the construction script logs". The shipped build log records
per-source edge and node counts and some "skipped unmapped" lines, but NOT duplicate merging or conflict handling. This
script therefore produces R-E3 in two parts and says which part each number comes from:

  A. parsed from logs/build_kg.log (and any other log passed with --logs): source-level nodes/edges added, lines that
     mention skipped / unmapped / dropped / duplicate / conflict entities;
  B. computed directly from the built graph (--kg, the shipped unified_kg.pt or the masked file):
       * duplicate edges: identical (head, relation, tail) repeated in an edge type;
       * conflicting edges: the same (head, tail) present under opposite regulation relations (upregulates vs
         downregulates) for the same node-type pair, e.g. drug up- and down-regulating the same gene; how many pairs, and
         per relation pair;
       * identifier systems per node type, inferred from raw_ids patterns (DrugBank DBxxxxx, DOID:, MONDO:, Entrez integer,
         Reactome R-HSA-, GO:, UBERON:, UMLS/MedDRA C-codes, ...) with the share of ids matching each pattern;
       * entities per type and isolated nodes (a proxy for entities that entered the graph without edges).
Numbers that cannot be derived from the log or the graph (entities matched across sources by shared identifiers, entities
dropped in mapping) are reported as "not logged" and listed in results/rerun/stats/R-E3_needed_from_builder.txt with the
exact counters to add to src/data/kg_builder.py so the next build records them.

Usage
  python -m exdrkg_rerun.data.kg_harmonisation --root . --kg data/processed/unified_kg.pt --logs logs/build_kg.log
"""
import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from exdrkg_rerun.common.torch_io import register_pyg_safe_globals; register_pyg_safe_globals()

ID_PATTERNS = [("DrugBank", r"^DB\d{5}$"), ("DOID", r"^DOID:\d+$"), ("MONDO", r"^MONDO:\d+$"), ("Entrez Gene", r"^\d+$"), ("Reactome", r"^R-[A-Z]{3}-\d+"),
               ("Gene Ontology", r"^GO:\d{7}$"), ("UBERON", r"^UBERON:\d+$"), ("UMLS/MedDRA C-code", r"^C\d{7}$"), ("ATC/pharm class", r"^N\d{4}|^[A-Z]\d{2}[A-Z]{2}\d{2}$"),
               ("UniProt", r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"), ("Symbol/name", r"^[A-Za-z][\w\- ,\(\)/']+$")]
OPPOSITE = [("upregulates", "downregulates")]
KEYWORDS = re.compile(r"(skipp|unmapp|dropp|discard|duplicate|conflict|merged|removed|excluded)", re.I)


def parse_logs(paths):
    rows = []
    for p in paths:
        for line in open(p, errors="ignore"):
            m = re.search(r"\|\s*(INFO|WARNING|DEBUG|SUCCESS|ERROR)\s*\|\s*(.*)$", line)
            msg = m.group(2).strip() if m else line.strip()
            if re.search(r"(nodes|edges|Drug features|Gene features)", msg) or KEYWORDS.search(msg): rows.append({"log": Path(p).name, "message": msg})
    return pd.DataFrame(rows)


def id_systems(kg):
    out = []
    for t in kg.node_types:
        ids = [str(x) for x in kg[t].raw_ids]; c = Counter()
        for x in ids:
            for name, rx in ID_PATTERNS:
                if re.match(rx, x): c[name] += 1; break
            else: c["unrecognised"] += 1
        n = len(ids); top = c.most_common(3)
        out.append({"node_type": t, "entities": n, "id_system(s)": "; ".join(f"{k} ({100 * v / n:.1f}%)" for k, v in top), "example_ids": ", ".join(ids[:3])})
    return pd.DataFrame(out)


def duplicates_and_conflicts(kg):
    dup_rows, seen = [], {}
    for et in kg.edge_types:
        ei = kg[et].edge_index.cpu().numpy(); key = ei[0].astype(np.int64) * (int(ei[1].max()) + 1 if ei.shape[1] else 1) + ei[1]
        n_dup = int(len(key) - len(np.unique(key))); dup_rows.append({"edge_type": "|".join(et), "edges": int(ei.shape[1]), "duplicate_edges": n_dup})
        seen[et] = set(zip(ei[0].tolist(), ei[1].tolist()))
    conf = []
    for (a, b) in OPPOSITE:
        for et in kg.edge_types:
            if et[1] == a:
                other = (et[0], b, et[2])
                if other in seen:
                    n = len(seen[et] & seen[other]); conf.append({"relation_pair": f"{et[0]} {a}/{b} {et[2]}", "pairs_in_both": n, "edges_up": len(seen[et]), "edges_down": len(seen[other]),
                                                                   "pct_of_smaller": round(100 * n / max(1, min(len(seen[et]), len(seen[other]))), 2)})
    return pd.DataFrame(dup_rows), pd.DataFrame(conf)


NEEDED = """Counters to add to src/data/kg_builder.py (log them per source and per node type) so R-E3 can be filled completely:
  1. n_matched[type][source]      entities matched to an existing node through a shared identifier (DrugBank id, Entrez id, DOID/MONDO, Reactome id)
  2. n_unmapped_dropped[type][source]   source records whose identifier could not be mapped and were dropped (write the ids to a TSV)
  3. n_duplicate_edges_merged     identical (head, relation, tail) contributed by two or more sources, merged into one edge (write source pairs)
  4. n_conflicts                  (head, tail) pairs with opposite relations, plus the rule applied (keep both / keep higher-confidence / drop) and how many kept or removed
  5. the identifier mapping tables actually used (file names and row counts)
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--kg", default="data/processed/unified_kg.pt")
    ap.add_argument("--logs", nargs="*", default=["logs/build_kg.log"]); ap.add_argument("--out", default="results/rerun/stats")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    kg = torch.load(root / a.kg, weights_only=False, map_location="cpu")
    logs = parse_logs([root / p for p in a.logs if (root / p).exists()]); dup, conf = duplicates_and_conflicts(kg); ids = id_systems(kg)
    logs.to_csv(out / "R-E3_log_lines.csv", index=False); dup.to_csv(out / "R-E3_duplicates_per_edge_type.csv", index=False)
    conf.to_csv(out / "R-E3_conflicts.csv", index=False); ids.to_csv(out / "R-E3_identifier_systems.csv", index=False)
    summary = {"duplicate_edges_total": int(dup.duplicate_edges.sum()), "edge_types_with_duplicates": int((dup.duplicate_edges > 0).sum()),
               "conflicting_pairs_total": int(conf.pairs_in_both.sum()) if len(conf) else 0,
               "matched_across_sources_by_shared_identifiers": "not logged", "entities_unmapped_and_dropped": "not logged (only 'skipped N unmapped' lines, see R-E3_log_lines.csv)",
               "conflict_resolution_rule": "not logged in the builder; opposite relations are stored as separate edge types (see R-E3_conflicts.csv)"}
    json.dump(summary, open(out / "R-E3_summary.json", "w"), indent=1); (out / "R-E3_needed_from_builder.txt").write_text(NEEDED)
    (out / "R-E3.md").write_text("## Identifier systems per node type\n" + ids.to_markdown(index=False) + "\n\n## Conflicting (up- vs down-regulation) pairs\n" +
                                 (conf.to_markdown(index=False) if len(conf) else "none found") + "\n\n## Duplicate edges\n" + dup[dup.duplicate_edges > 0].to_markdown(index=False) + "\n\n## Log lines\n" + logs.to_markdown(index=False) + "\n\n" + NEEDED)
    print(json.dumps(summary, indent=1)); print(ids.to_string(index=False)); print(conf.to_string(index=False) if len(conf) else "no conflicts")


if __name__ == "__main__":
    main()
