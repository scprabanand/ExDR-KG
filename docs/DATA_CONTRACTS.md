# Input and output file formats

**Score file** `results/rerun/main/scores/<Model>.csv`: `drug_id, disease_id, label, score`. One row per test pair, identical pair set and order for every model (`evaluation.main_evaluation` validates this by pair key).

**Split folder** `data/splits/<split>/{train,val,test}.csv`: `drug_id, disease_id, label` (created by `data.rebuild_splits`; `<split>` = `standard_degree_matched`, `standard_random`, `zero_shot`, `temporal`). The temporal test file needs a first-evidence-year column for Steps 7 and 14.

**LLM pair file** `data/llm/pairs.jsonl` / `test_pairs.jsonl`: `{"drug_id","disease_id","drug","disease","label","split","paths":[...],"passages":[...],"entities":[...],"template":"..."}` (paths from the retriever on the MASKED graph; passages from the original RAG pipeline).

**Evidence file** `results/rerun/<sets>/evidence.jsonl` (from `explainability.build_evidence`): `{"pair_id","drug","disease","paths":[...],"entities":[...]}`.

**Explanation file** `explanations.jsonl`: `{"model","set","pair_id","explanation"}` (`pair_id` = `drug_id|disease_id`).

**Disease-family map** `data/disease_family.csv`: `disease_id, family`. **Rater roster** `data/raters.csv`: `rater_id, role (pharm|clinician|senior), took_part_in_rlhf (0/1)`.

**Case literature** `data/case_literature.csv`: `drug_id, disease_id, drug_name, disease_name, mondo_id, reference_type, reference_id, reference_year, mechanism_note, failure_reason`.

**RAG configuration** `data/rag_config.json`: see `configs/rag_config.example.json`; null fields print as NOT RECORDED in Table R-K.

**Ablation manifest** `configs/ablation_manifest.yaml`: see the example; one entry per variant V1-V5 with checkpoint paths.

**Outputs** all under `results/rerun/<topic>/`: CSV (machine-readable), MD (readable), TEX (LaTeX), JSON (full detail), plus `logs/rerun/*.log` from the driver.
