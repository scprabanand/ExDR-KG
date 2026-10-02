# Step 1 - Freeze the setup (standard split), computed 2026-09-29

## 1. Hashes (md5)

| Item | md5 |
|---|---|
| Code commit (repository HEAD on GitHub at time of check) | cc0b959 (full hash on the repo; the data commit is 7ca0515) |
| data/processed/unified_kg.pt (shipped, NOT masked) | 6bad37b1f1a3095b0b110b4ff45baca3 |
| masked_standard.pt (built by rerun/masked_kg.py in this session; torch serialization is not byte-stable across versions, so re-hash on your machine) | 1dd481ac0460913eb6d35383c4ee511b |
| standard/train.csv | cbda39f967620a179aece8108c82dc14 |
| standard/val.csv | 353c7832ac4f526b4aeaa217d04c4010 |
| standard/test.csv | 2a7f7cdade74b06a56b6a99b4d110048 |
| zero_shot/train, val, test | 0e663bebb6179115789346cb98fe5040, 013e2151c567de718fae2fc3ed074d1d, 1bd838cbe235b999ed233f3f906e1e5a |
| temporal/train.csv | 2faf8188cf4e830909d86b853bcef467 |
| txgnn/train, val, test | 08a931f03848325c7c533bc84465f90e, 4622ad69234ddd865f2b63aec03bf63f, d054abd2720b0b6d37251d451a3daae6 |
| Fixed negative pool (114,400 rows; sorted part/drug/disease) | 5d87e6fcbc0b552b40fe1d7f48cf387e |

The negative pool is not stored as its own file. It is the label-0 rows of the three split CSVs (train 91,500, val 11,400, test 11,500; exactly 100 per positive).

## 2. Masking confirmation

- Held-out positives: 229 (114 validation + 115 test). Directed edges removed: 458 (each pair in treats/palliates and the reverse).
- Verification (assertion in `masked_kg.py`): 0 held-out pairs remain in any drug-disease edge type. Train positives kept in the graph: 915 of 915.
- Source-equivalent duplicates: none exist in this graph. Only two indication relations exist (treats 755, palliates 389), no pair is in both (0), and the 1,144 indication pairs equal the 1,144 split positives exactly. There is no separate DrugBank indication relation, so the "overlapping DrugBank and Hetionet annotations" described in Section 4.1 do not exist in this graph.
- The shipped unified_kg.pt is not masked. Every encoder run must load the masked file.

## 3. Seed and environment

- Seed 42 is set inside every rerun script (bootstrap, sampling, torch, numpy).
- Original run environment (GPU, CUDA, PyTorch): not recorded anywhere in the repository. The manuscript says A100 80 GB, CUDA 11.8, PyTorch 2.0; there is no log to confirm it.
- Environment of this check: Python 3.11.15, PyTorch 2.14.0, CUDA build 13.0, no GPU.

## 4. Findings the freeze exposed

1. **Negative pairs are shared across splits.** 4,674 of the 11,104 unique test negative pairs (42%) also appear as training negatives (4,825 train/val, 876 val/test). Positives never overlap. A model that memorises training pairs can reject part of the test negatives for free, so headline metrics are inflated for any method that learns pair identity.
2. **Duplicate rows inside splits.** Repeated (drug, disease) rows: train 22,034, val 416, test 396. These are repeated negatives (train has 69,466 unique negatives in 91,500 rows; test 11,104 in 11,500).
3. **Negatives are random corruptions**, not type-constrained: corrupt_head 5,750 and corrupt_tail 5,750 in test. Section 4.1 of the manuscript says otherwise.
4. **Counts:** the standard split has 115 test positives, not the 116 in the plan. The manuscript's 115,544 total pairs equals the sum of the three standard splits (92,415 + 11,514 + 11,615), so that figure is consistent; the 500-disease zero-shot claim still is not (270 diseases exist).
5. **No false negatives:** none of the 114,400 negatives is a known indication in the knowledge graph.

## Table R-K (part b): rows filled by Step 1

| Item | Value |
|---|---|
| Code commit hash and data file hashes | see section 1 |
| GPU, CUDA, PyTorch versions | not recorded for the original runs; rerun to record |
| Entity harmonisation / conflict rule | conflicting-edge handling is not logged; no duplicate indication edges exist |
