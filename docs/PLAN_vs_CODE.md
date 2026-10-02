# ExDR-KG_Rerun_Plan.docx versus the rerun project

**Answer: almost, not exactly.** Every plan step (1 to 14) and every result table (R-A to R-K) now has code. In this check I found 11 gaps between the plan and the kit and fixed them; 9 differences remain that need a decision or your input (section 3). Nothing was executed.

Status words: **Match** = the code does what the plan asks; **Fixed** = differed from the plan, corrected in this check; **Input** = correct, but needs a file or facts only you have; **Deviation** = differs from the plan on purpose or because the repository cannot do it.

## 1. Step by step

| Step | Plan asks | Kit (module) | Status |
|---|---|---|---|
| 1 | Commit hash, md5 of masked HKG / splits / negative pool, mask check, seed 42, GPU/CUDA/PyTorch into R-K | `data.masked_kg`, driver `splits` (writes `R-K_freeze_<split>.json`, `gpu_info.csv`, `pip_freeze.txt`) | Match |
| 2 | R-A: 8 models, scores CSV per model, pooled AUROC/AUPRC, Hit@1/10, MRR (one definition), CIs stratified by disease family, DeLong + paired bootstrap, Bonferroni 7 (0.0071), random AUPRC 0.010, AUPRC sanity flag | `evaluation.main_evaluation` + scorers | Match. Family stratification **Input** (`data/disease_family.csv`) |
| 2 | GPT-4+RAG = the submitted model (GPT-4-turbo-2024-04-09) | driver used the new model string | **Fixed**: `LEGACY_API_MODEL` for Steps 2 and 4, `API_MODEL` only for Step 11 |
| 2 | Headline numbers on the standard split, 11,615 pairs (116 pos / 11,499 neg) | driver defaulted to rebuilt negatives (100:1 gives 11,716 pairs) | **Fixed**: default `STRATEGY=original`; rebuilt negatives stay as a sensitivity run |
| 3 | R-B, successive tests with Bonferroni 4 (0.0125), PPO/LoRA yes-no | `evaluation.ablation_evaluation`, `models.check_ppo_lora_sharing` | Match. Checkpoints and manifest **Input** |
| 4 | R-D: H1 (200 pairs), H2 defined and listed, BioBERT NER, k/N, path faithfulness per entity, six models | `explainability.*` | Match. H1 pair IDs, entity vocabulary, TxGNN paths **Input** |
| 5 | R-E1 (igraph, 10,000 pairs, seed 42), R-E2 for 12 types, log-log degree figure, R-E3 from build logs | `graph_analysis.hkg_statistics`, `data.kg_harmonisation` | Match for E1/E2. R-E3 **Partial**: the builder logs do not contain matches, drops or a conflict rule; the script lists the counters to add |
| 6 | R-E4: paths <= 4 hops on masked HKG, k (%) with path, median candidate paths, mean shortest length; test pos/neg, zero-shot pos, temporal pos | `graph_analysis.path_availability` | Match (**Deviation**: reproduces the retriever's DFS, the plan says BFS; same path set). Temporal row needs a rebuilt temporal test file with dates |
| 7 | R-F: 5 cases + 1-2 failures, MONDO id, absence check, top path, Level 2 explanation, reference, "mechanism agrees?" | `explainability.case_studies` | Match; **Fixed**: added the `mechanism_agrees` column. Literature rows **Input** |
| 8 | R-I part a: pairs, overlap k (%), experts RLHF/eval/both, comparisons, raters per comparison, **agreement on preference labels (%)** | `human_eval.rlhf_audit` | **Fixed**: agreement row added (needs a preference column); outputs renamed `R-I_part_a.*` (were `R-H`) |
| 9 | L = 2, 4, 8, 10, everything else identical, L=6 not retrained and equal to V1, MAD 5,000 nodes seed 42, per-layer MAD, params/time/memory | `models.depth_ablation`, `evaluation.depth_summary` | Match; **Fixed**: "parameters, time, memory must rise with L" check added. **Deviation**: Stage 1 here is one full-graph step per epoch, the plan says batch 256 (see 3.3) |
| 10 | Two alternative backbones, identical Stage-2 settings, R-G with H1 k/200 and Stage-2 hours, **BioMistral row = V4 in R-B** | `models.stage2_finetune`, `models.score_stage2`, `evaluation.backbone_summary` | **Fixed**: BioMistral row can now be the existing V4 (`--external-ref-scores`, used automatically when the Step 3 V4 scores exist) |
| 11 | R-H: existing GPT-4 row + 2 new rows, context given, AUROC/AUPRC/Hit@10, **H1 k/200**, temperature 0, same sample for all models, RAG config | `baselines.llm_api`, `evaluation.llm_baseline_summary` | **Fixed**: added the Context column, the existing-model row, and the H1 hallucination column (driver now generates and checks explanations for the new rows) |
| 12 | 100 new pairs by disease family, >= 3 blind raters, rubric with anchors, free text, alpha per dimension, adjudication at >= 2 points, themes; R-I b, R-J a and b | `human_eval.rating_packets`, `rating_analysis` | Match; **Fixed**: four systems supported (the plan's R-I b has four rows; blinding was limited to three), old 200-pair composition column for R-J a, theme table with the plan's five rows and example quotes for R-J b (outputs renamed `R-J_b_*`) |
| 13 | Facts sheet R-K (a: panel profile, b: protocol facts) | `reporting.facts_sheet` | **Fixed**: new; auto-fills hashes, versions, GPU, rubric anchors, allocation, blinding, RAG config; marks the rest `TO FILL` |
| 14 | Figure 5 legend below, 600 dpi; Figures 15/17 -> 11/13 full width, >= 9 pt, split into panels, new Step 7 cases; degree figure; language checks and acronyms | `reporting.figures`, `reporting.language_audit` | Match. Figure 5 **Input** (per-year scores of the temporal models) |

## 2. Consistency rules of the plan (Section 1)

| Rule | Status |
|---|---|
| One split for headline numbers (test set, 11,615 pairs) | Match after the fix (`STRATEGY=original`) |
| ExDR-KG in R-A = V5 in R-B | Match (`ablation_evaluation` copies V5 to the main scores and checks equality) |
| V1 in R-B = L=6 in R-C | Match (L=6 metrics are taken from the V1 scores) |
| BioMistral in R-G = V4 in R-B | Match after the fix |
| Counts are whole numbers, k / N reproduces the percentage | Match for hallucination, path availability, overlap (asserted in `hallucination_eval`) |
| Parameters, time, memory rise with L | Match after the fix |
| Metric definitions fixed once | Match: pooled AUROC/AUPRC; Hit@k/MRR per disease, ties share the average rank. Candidates = drugs paired with the disease in the test file (plan: "all candidate drugs"); write this into the manuscript |
| Seed 42, same commit, same masked HKG, same negative pool | Match (seed 42 in every module; commit hash in the freeze file) |

## 3. Differences that remain (decisions or input needed)

1. **ExDR-KG "scoring pass (Eq. 22)"**: the repository has no 0.8 / 0.2 fusion, so the R-A ExDR-KG row will score with the model as implemented. Implement Eq. 22 or drop it from the paper.
2. **Checkpoints**: the plan says "load the final checkpoint of each model". The folder has no trained checkpoints (Step 2 and Step 3 need V1-V5, the encoder, the retriever, TxGNN); they must come from the machine that produced the manuscript numbers, or the models must be retrained.
3. **Stage 1 batch**: plan says batch 256 for all depths; the repository's config says 32 and `pretrain_step` is full-graph. Check the original Stage-1 loop and use the same one for all depths.
4. **Negatives**: the shipped negatives are shared across splits and are random head/tail corruptions, not type-constrained as the manuscript says. The plan forbids changing the negative pool; the kit therefore defaults to the original one and offers the rebuilt one as a second run. Decide which goes in the paper and say so.
5. **R-E3 harmonisation counts**: the logs do not contain them (see the counters listed in `results/rerun/stats/R-E3_needed_from_builder.txt`).
6. **Temporal split**: no test file with dates; Steps 6 (temporal row), 7 and Figure 5 need it.
7. **Step 12 raters and family map**: `data/disease_family.csv`, `data/raters.csv` and the RLHF pair list are needed (DOID-based diseases have no family labels in the repository).
8. **Step 11 RAG configuration**: to be copied from the original pipeline; unknown fields print as NOT RECORDED.
9. **Eq. 11 questions in R-K part b** (edge weight w_r, dimensions of z_p and z_dq): the repository's path scorer is an MLP over [mean node embedding, edge-type diversity, pathway coverage, semantic coherence]; it has no w_r product and no length penalty. `facts_sheet` records this as a code fact; the answer to the reviewer must match it or the code must change.

Order: settle 1-4, place the checkpoints (2), then run the stages in the order listed by `bash rerun/scripts/run_pipeline.sh help`.
