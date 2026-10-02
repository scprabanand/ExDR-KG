# ExDR-KG reviewer rerun

Modular code for the experiments the reviewers asked for (rerun plan, Steps 1-14). **Nothing here has been executed end to end.** Every script writes its own outputs; no result number is typed into any file. Read `docs/REPOSITORY_ISSUES.md` first: it lists the defects found in the main repository that this code works around (unmasked shipped graph, non-differentiable reasoner loss, shared negatives, synthetic source data).

## Layout

```
rerun/
├── README.md                     this file
├── pyproject.toml, requirements.txt, Makefile, .gitignore
├── configs/                      pipeline.env.example, ablation_manifest.example.yaml, rag_config.example.json
├── docs/                         STEP_MAP.md (plan step -> module -> table), PLAN_vs_CODE.md, MANUSCRIPT_vs_CODE.md, OUTPUT_LAYOUT.md, DATA_CONTRACTS.md (file formats),
│                                 REPOSITORY_ISSUES.md, R-K_step1_results.md (measured Step-1 facts)
├── scripts/                      run_pipeline.sh (stage driver), init_workspace.sh (create folders), collect_reports.py (copy tables/figures to reports/)
├── repo_layout/                  browsable copy of the input/output folder tree (data, models, logs, results, reports); see docs/OUTPUT_LAYOUT.md
├── tests/                        unit tests for the pure-logic parts (no GPU, no data needed)
└── exdrkg_rerun/                 the package
    ├── common/                   metrics (AUROC, AUPRC, Hit@k, MRR, bootstrap, DeLong), stats (Krippendorff alpha), torch_io
    ├── data/                     rebuild_splits (clean negatives), masked_kg (leak-free HKG + hashes), kg_harmonisation
    ├── baselines/                graph_features, graph_baselines, gnn_baselines (R-GCN/CompGCN/HAN/Metapath2vec), txgnn, biogpt, llm_api
    ├── models/                   stage2_finetune, score_stage2, score_exdrkg, score_ablation_variants, check_ppo_lora_sharing, depth_ablation
    ├── evaluation/               main_evaluation (R-A), ablation_evaluation (R-B), depth_summary (R-C), backbone_summary (R-G), llm_baseline_summary (R-H/R-K)
    ├── explainability/           build_eval_sets, build_evidence, generate_explanations, hallucination_eval (R-D), case_studies (R-F)
    ├── graph_analysis/           hkg_statistics (R-E1/R-E2), path_availability (R-E4)
    ├── human_eval/               rlhf_audit (R-I part a), rating_packets + rating_analysis (blinded expert study, R-I part b, R-J a/b), plus reporting.facts_sheet (R-K)
    └── reporting/                figures (600 dpi), language_audit (manuscript .docx)
```

Conventions: every module is runnable as `python -m exdrkg_rerun.<package>.<module> --help`, takes `--root` (the repository root), seeds with 42, reads and writes only under `data/`, `models/`, `results/rerun/` and `logs/rerun/`, and stops with a clear message instead of inventing missing inputs.

## Install and run

Copy this folder to `<repository>/rerun/` and work from the repository root:

```bash
pip install -r rerun/requirements.txt          # or: pip install -e rerun
export PYTHONPATH="$PWD/rerun:$PYTHONPATH"      # not needed after pip install -e
bash rerun/scripts/init_workspace.sh                  # create data/, models/, logs/, results/, reports/ folders (docs/OUTPUT_LAYOUT.md)
cp rerun/configs/pipeline.env.example rerun/configs/pipeline.env   # edit SPLIT, STRATEGY, API_MODEL, ...
bash rerun/scripts/run_pipeline.sh help         # list the stages
bash rerun/scripts/run_pipeline.sh splits       # Step 1: run this first
```

Stage order (see `docs/STEP_MAP.md` for the table each stage fills): `splits` -> `graph-baselines` -> `main-eval` -> `ablation` -> `hallucination` -> `hkg-stats` -> `paths` -> `cases` -> `rlhf-audit` -> `depth` -> `backbones` -> `llm-baselines` -> `facts` -> `figures` -> `language`. `make -f rerun/Makefile help` lists the same stages.

## Step 12 (blinded expert evaluation, human process)

```bash
python -m exdrkg_rerun.human_eval.rating_packets sample  --root . --test <test.csv> --family-map data/disease_family.csv --exclude data/rlhf/pairs.txt data/expert_pairs_200.txt
# build evidence and explanations for the sampled pairs (explainability.build_evidence / generate_explanations with --sets results/rerun/expert12)
python -m exdrkg_rerun.human_eval.rating_packets packets --root . --explanations <jsonl> --models A B C --roster data/raters.csv --show-paths no --literature-access no
python -m exdrkg_rerun.human_eval.rating_analysis collect --root .     # after raters return their sheets
python -m exdrkg_rerun.human_eval.rating_analysis analyse --root . --adjudication ... --comments ... --roster ...
```

## Figures

`reporting.figures fig5` (temporal AUROC per year; needs per-year scores or `--auroc-csv`), `fig11` (qualitative comparison of one Step 7 case), `fig13` (one full-width figure per case), `degree` (supplementary). All are drawn at final printed width with a minimum 9 pt font and refuse to save otherwise.

## Development

`make -f rerun/Makefile test` (from the repository root) runs the unit tests (numpy/pandas/scipy only; tests needing optional libraries are skipped). Add new experiments as a module in the matching package and register a stage in `scripts/run_pipeline.sh` and a row in `docs/STEP_MAP.md`.
