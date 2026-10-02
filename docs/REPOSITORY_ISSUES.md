# Problems found in the repository and Method notes

## Problems found in the repository that this code works around

1. `data/processed/unified_kg.pt` still contains every indication edge, including validation and test positives. Any encoder trained on it sees the labels. Always build the masked graph with `masked_kg.py`.
2. `src/models/llm_reasoner.py` builds its prediction loss from a score parsed out of generated text, so no gradient reaches the LoRA weights. `stage2_finetune.py` uses a score head on a `<SCORE>` token instead.
3. `scripts/ablation_study.py` (main repository) scores with `torch.rand_like`, and `ablation_visualize.py` and `generate_figures.py` contain typed-in numbers. Do not use their outputs.
4. Test negatives in the standard split are random head/tail corruptions, not type-constrained as the manuscript states. Shortest-path distance alone reaches about 0.805 AUROC on that test set, so path-based methods are easier than they look. Consider hard negatives (same-type, degree-matched) for a second evaluation.
5. `data/raw/drugbank` and `data/raw/disgenet` appear to be the synthetic files from `create_synthetic_data.py` (232 KB and 32 KB). Replace them with the real releases before any reported run.
6. The temporal split has no validation/test files and its years are estimated. Rebuild it from real approval dates before Step 7.

## Method notes

- Hit@k and MRR: per test disease with at least one positive, candidates are the drugs paired with that disease in the test file; ties share the average rank.
- Bootstrap CIs are stratified by label, not by disease family, because disease-family labels are not available for DOID identifiers in this repository.
- `baselines.graph_baselines` baselines: random, shortest path, Adamic-Adar 2-hop and 3-hop, and gradient boosting on path statistics with and without collaborative features from known training indications. Training positives are hidden fold by fold so a training pair never sees its own edge. These are stronger, honest reference points; they are not R-GCN, CompGCN, HAN, TxGNN or Metapath2vec.
