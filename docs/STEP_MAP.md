# Rerun plan step -> module -> table

| Step | Purpose | Module(s) (`python -m exdrkg_rerun...`) | Table / output | Needs |
|---|---|---|---|---|
| 1 | Freeze setup, clean negatives, masked HKG, hashes | `data.rebuild_splits`, `data.masked_kg` | R-K (hashes, versions), `docs/R-K_step1_results.md` | CPU |
| - | Leakage-safe reference baselines | `baselines.graph_baselines` (uses `baselines.graph_features`) | reference rows | CPU |
| 2 | Main test-set evaluation | `baselines.gnn_baselines`, `baselines.txgnn`, `baselines.biogpt`, `baselines.llm_api`, `models.score_exdrkg`, `evaluation.main_evaluation` | R-A | GPU |
| 3 | Ablation V1-V5 | `models.score_ablation_variants`, `evaluation.ablation_evaluation`, `models.check_ppo_lora_sharing` | R-B, PPO/LoRA answer | GPU |
| 4 | Hallucination and faithfulness | `explainability.build_eval_sets`, `build_evidence`, `generate_explanations`, `hallucination_eval` | R-D | GPU |
| 5 | HKG statistics, harmonisation | `graph_analysis.hkg_statistics`, `data.kg_harmonisation` | R-E1, R-E2, R-E3, degree figure | CPU |
| 6 | Path availability + leakage check | `graph_analysis.path_availability` | R-E4 | CPU (numba) |
| 7 | Temporally novel case studies | `explainability.case_studies` | R-F | CPU |
| 8 | RLHF vs evaluation audit | `human_eval.rlhf_audit` | R-I (part a) | CPU |
| 9 | Encoder depth ablation, MAD | `models.depth_ablation`, `evaluation.depth_summary` | R-C | GPU |
| 10 | LLM backbone replacement | `models.stage2_finetune`, `models.score_stage2`, `evaluation.backbone_summary` | R-G | GPU |
| 11 | Updated LLM baselines | `baselines.llm_api`, `evaluation.llm_baseline_summary` | R-H, R-K | API |
| 12 | Independent blinded expert study | `human_eval.rating_packets`, `human_eval.rating_analysis` | R-I (part b), R-J (a, b) | human raters |
| 14 | Figures and language | `reporting.figures`, `reporting.language_audit` | figures, language report | CPU |
| 13 | Facts sheet | `reporting.facts_sheet` (auto-fills what the files contain, marks the rest TO FILL) | R-K (a, b) | your records |
