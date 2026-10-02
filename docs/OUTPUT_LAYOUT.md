# Where everything is saved

Same conventions as the ExDR-KG repository (`data/`, `models/`, `logs/`, `results/`, `reports/`). Create all folders once, from the repository root:

```bash
bash rerun/scripts/init_workspace.sh
```

The full tree is in `configs/workspace_dirs.txt` (and as a browsable copy in `repo_layout/`). Everything the rerun code writes goes under `results/rerun/`, `logs/rerun/`, `models/` and `checkpoints/rerun/`; nothing is written into `src/` or into the existing `results/ablation_study` or `results/figures` folders, so the original outputs stay untouched.

| What | Folder | Examples |
|---|---|---|
| **Logs** (one file per stage/run) | `logs/rerun/` | `rebuild_splits.log`, `depth_L2.log`, `step7.log` |
| **Tables** (CSV + Markdown + LaTeX per table) | `results/rerun/<topic>/R-*.csv .md .tex` | `main/R-A.*`, `ablation/R-B.*`, `hallucination/R-D.*`, `paths/R-E4.*`, `stats/R-E2_*.*`, `cases/R-F.*`, `backbones/R-G.*`, `llm/R-H.*`, `llm/R-K.md`, `audit/R-I_part_a.md` |
| **Figures** (600 dpi PNG + PDF) | `results/rerun/figures/` and `results/rerun/stats/` | `fig05_temporal_generalisation`, `fig11_qualitative_comparison`, `fig13a_case_1`, `figS1_degree_distribution`, `stats/fig_degree_distribution` |
| **Per-model score files** | `results/rerun/main/scores/<Model>.csv` | one row per test pair; used by every table |
| **Depth ablation** | `results/rerun/<split>/` | `depth_L2.json` ... `depth_L10.json`, `R-C.*`, `fig_mad_vs_depth.png` |
| **Path availability and leakage check** | `results/rerun/paths/` | `pairs_*.csv`, `R-E4.*`, `leakage_check.csv` |
| **Expert study (Step 12)** | `results/rerun/expert12/` | `rater_sheets/` (send to raters), `rater_sheets_filled/` (returned), `blinding_key.csv` (keep private), `R-I_*.csv` |
| **Language report** | `results/rerun/language/` | `audit_summary.md`, `grammar_suggestions.csv`, `apply_report.csv` |
| **Environment record** | `results/rerun/` | `pip_freeze.txt`, `gpu_info.csv`, `R-K_freeze_<split>.json` |
| **Trained models** | `models/stage2/`, `models/ablation/v1..v5/`, `checkpoints/rerun/` | LoRA adapters + `run.json`, retrained encoders |
| **Rebuilt splits** | `data/splits/<split>_<strategy>/` | `train.csv`, `val.csv`, `test.csv`, `split_stats.json` |
| **Masked graphs** | `data/processed/masked_<split>.pt` | leak-free training HKG |
| **Manuscript-ready copies** | `reports/tables/`, `reports/figures/` | filled by `python rerun/scripts/collect_reports.py` |

Git: add `results/rerun/`, `logs/rerun/`, `models/`, `checkpoints/` and `data/processed/` to the repository `.gitignore` (large or regenerable); keep `reports/` under version control. `results/rerun/expert12/blinding_key.csv` and rater sheets must never be committed together with model names.
