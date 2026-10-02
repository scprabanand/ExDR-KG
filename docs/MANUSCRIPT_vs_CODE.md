# Manuscript_Revised_V1.docx versus the rerun project and the repository

**Answer: no, they do not match exactly.** Most headline hyperparameters agree, but there are (A) places where the repository or the rerun code does something different from what the manuscript states, (B) statements the code cannot support, (C) contradictions inside the manuscript itself, and (D) manuscript content the rerun kit does not yet produce. Checked against the manuscript text, `configs/default.yaml`, `src/models/*.py` and every module in this kit. Nothing was executed.

## 1. Matches

| Item | Manuscript | Code |
|---|---|---|
| Graph size | 50,561 nodes, 12 node types, 56 relation types, 4,545,138 edges (Sec. 3.2, 4.1, 4.2) | measured in Step 1 (`docs/R-K_step1_results.md`); Table 3 edge column sums to 4,545,138 |
| Split sizes | 92,415 / 11,514 / 11,615 = 115,544 pairs; 100 negatives per positive | `data.rebuild_splits` keeps positives and the 100:1 ratio |
| Encoder | 6 layers, 8 heads, d_model 256, lr 1e-4, weight decay 1e-5, 100 epochs, mask 15% | `configs/default.yaml`, `models.depth_ablation` (L=6) |
| Retriever | max 4 hops, top-K 10, MMR lambda 0.3 | `default.yaml` |
| LoRA / Stage 2 | rank 16, alpha 32, lr 2e-5, 10 epochs, mu = 0.5 | `models.stage2_finetune` defaults |
| PPO | lr 1e-6, KL 0.1, 3 epochs | `default.yaml` |
| Bootstrap | 1,000 resamples | `common.metrics` (seed 42) |

## 2. Mismatches between the manuscript and the repository / rerun code

| # | Manuscript says | Repository / rerun code does | Suggested action |
|---|---|---|---|
| 1 | Faithfulness loss (Eq. 17): cosine distance between decoder hidden state and the nearest evidence-entity embedding at NER-tagged positions (spaCy-BioBERT NER applied offline) | Repo: regex `EntityExtractor`, loss built from Python floats (no gradient). Rerun `stage2_finetune`: -log(1 - probability mass on entity first-tokens absent from the paths) | Either implement Eq. 17 in `stage2_finetune`, or rewrite Eq. 17 to the loss actually used; do not leave both |
| 2 | Total loss Eq. 18: L_cls + L_gen + mu L_faith (LM weight 1) | Repo: prediction + lambda_faith x faithfulness + lambda_coherence (0.3) x coherence. Rerun: BCE + 0.3 x LM + 0.5 x faith | Align: set `--lambda-lm 1.0` or change Eq. 18 |
| 3 | Score token: Xavier-uniform embedding, kept trainable | Rerun: mean-initialised, not trained (so it can be rebuilt at scoring time) | Disclose or add `modules_to_save` and save the embedding row |
| 4 | Final score Eq. 22: S_final = 0.8 s_dq + 0.2 s_KG | No `alpha` fusion found in `src/models/exdr_kg.py`; `score_stage2` scores with the head only | Implement fusion in scoring, or remove Eq. 22 |
| 5 | Reward model trained on 200 pairs, RLHF stage in the pipeline (Stage 3) | `default.yaml`: `use_reward_model: false`, `expert_annotations: 500`; stages listed as encoder / retriever / joint reasoning | Reconcile config and text; state which run produced V5 |
| 6 | Batch size 256, chunked edges of 100,000 | `default.yaml` `training.batch_size: 32`; `pretrain_step` is a full-graph step; depth ablation runs one such step per "epoch" | Check the repo's actual Stage-1 loop; report the real batch definition |
| 7 | Retriever enumerates paths of "at most 4 **intermediate nodes**" by constrained **breadth-first** search | Config max_hops = 4 (edges, i.e. 3 intermediate nodes); enumeration is a directed DFS | Pick one definition; Step 6 counts <= 4 hops |
| 8 | Negatives are "type-constrained" | Shipped negatives are random head/tail corruptions (Step 1 finding); 42% of test negatives also appear as train negatives | Use the rebuilt splits and describe them exactly |
| 9 | Drug features: 2,048-bit ECFP-4 (Sec. 3.2.1) but "pseudo-Morgan fingerprints" (Sec. 4.1); disease: Poincare embedding of MONDO | DrugBank and DisGeNET files in the repo are synthetic (`create_synthetic_data.py`); features were not verified | Replace with real releases before reporting; fix the wording |
| 10 | LoRA trainable parameters ~18.9M (0.26%) | Attention-only rank-16 LoRA on Mistral-7B is 13.6M; the run record of `stage2_finetune` (`run.json`) will show the true count | Report the measured number |
| 11 | Baselines scored by "prompting for a calibrated probability, following Eq. 15" | Rerun BioGPT: yes/no logit score; GPT models: JSON probability | Describe the scoring used (stated in each script header) |
| 12 | Significance by Wilcoxon signed-rank on pairwise comparisons; CIs stratified by disease family | Rerun: DeLong (AUROC) and paired bootstrap; stratified by label unless `--family-map` is given | Update Sec. 5.1 / 5.14; supply `data/disease_family.csv` |
| 13 | CUDA 11.8, PyTorch 2.0 | `requirements.txt`: torch >= 2.1 | Record actual versions in Table R-K (`pip_freeze.txt`) |
| 14 | Zero-shot reported as Hit@20 | `common.metrics` had Hit@1/5/10 only | **Fixed now:** Hit@20 is computed |

## 3. Contradictions inside the manuscript (the rerun will replace these numbers)

- ExDR-KG AUROC is 0.856 in Table 5 but ablation V5 (the same full model) is 0.950 (Table 6) and validation AUROC is 0.947 (Table 9). Step 3 checks that V5 equals ExDR-KG.
- Table 6 V1 (encoder + MLP) is 0.870, higher than every graph baseline in Table 5 (0.645-0.723), which needs an explanation.
- Hallucination: ExDR-KG 4.5% (Table 5, Sec. 5.4), 3.2% (Fig. 9), 5% (V4); BioGPT 35.0% vs 34.1%; GPT-4+RAG 18.0% vs 15.2%; TxGNN 10.0% vs 8.7%.
- "Table 5" is used for both overall performance and expert ratings. "Six baselines" (Sec. 4.3) versus "seven baselines" and Metapath2vec in Table 5 (Sec. 5.1). Sec. 5.5 cites "Section 4.6", which does not exist.
- "500 rare diseases" (zero-shot, conclusion) versus 270 disease nodes in the graph (Sec. 4.2).
- Table 3 node counts add up to 60,160 (47,031 + 1,430 + 9,145 + 2,554) versus 50,561 nodes; round edge counts (840,000; 628,000; 512,000) look estimated.
- 200 pairs are described as both the RLHF training set (Sec. 3.6.1) and the evaluation set (Sec. 5.4), by the same 10 experts (Step 8 audit).

## 4. Manuscript content the rerun kit does not produce yet

- Table 9 hyperparameter sensitivity (K, MMR lambda, LoRA rank): no script.
- Figures 6-8 (radar, convergence, t-SNE) and the 5.10 heatmap are not regenerated; the reviewers asked to drop Figures 11-14.
- TxGNN-protocol split (`data/splits/txgnn`) is not part of any stage.
- Per-family zero-shot breakdown needs `data/disease_family.csv`.
- Inference latency (45 s per pair) is not measured.

## 5. Order to resolve

1. Decide items 1-4 (loss, weights, score token, score fusion): change either the code or the equations, then retrain.
2. Rebuild splits and rerun Steps 1-3; use the measured numbers to replace Tables 5 and 6, removing the contradictions in section 3.
3. Rerun Steps 4, 8 and 12 for hallucination and expert results.
4. Update the text (items 7-13) and the R-K table.
