#!/usr/bin/env bash
# ExDR-KG reviewer rerun: pipeline driver for the lab A100 server. Run from the REPOSITORY ROOT (the folder that contains src/, data/, rerun/).
#   bash rerun/scripts/run_pipeline.sh <stage>          (see `bash rerun/scripts/run_pipeline.sh help`)
# Settings: copy rerun/configs/pipeline.env.example to rerun/configs/pipeline.env and edit; it is sourced automatically.
# Every stage logs to logs/rerun/ and writes results under results/rerun/. Seed 42 everywhere. Nothing here was executed end to end by the author.
#
# stage            rerun-plan step   what it does                                                       hardware
# splits           1 (+ negatives)   rebuild leak-free negatives, masked HKG, hashes, environment       CPU
# graph-baselines  extra             leakage-safe graph-only reference baselines                        CPU
# main-eval        2                 score all models, build Table R-A                                  GPU
# ablation         3                 score V1-V5, Table R-B, PPO/LoRA sharing check                     GPU
# hallucination    4                 eval sets, evidence, explanations, Table R-D                       GPU
# hkg-stats        5                 connectivity, degrees, harmonisation (R-E1..R-E3), degree figure   CPU
# paths            6                 path availability per test pair (R-E4) + leakage check             CPU
# cases            7                 temporally novel case studies (R-F)                                CPU
# rlhf-audit       8                 RLHF vs evaluation overlap audit (R-I a)                           CPU
# depth            9                 encoder depth ablation with MAD (R-C)                              GPU
# backbones        10                LLM backbone replacement, Stage 2 only (R-G)                       GPU
# llm-baselines    11                updated LLM baselines: RAG and ExDR-KG paths (R-H, R-K)            API
# stage2           10 (helper)       Stage-2 LoRA runs for the backbones and V3                         GPU
# facts            13                R-K facts sheet: auto-fill from other steps, mark the rest TO FILL   CPU
# figures          14                figures redrawn at 600 dpi                                         CPU
# language         14                language audit of the manuscript (.docx)                           CPU
# (Step 12, the blinded expert evaluation, is a human process: see rerun/README.md for its four commands.)
set -euo pipefail
ROOT="${ROOT:-$(pwd)}"; cd "$ROOT"
[ -f rerun/configs/pipeline.env ] && source rerun/configs/pipeline.env
export PYTHONPATH="$ROOT/rerun:${PYTHONPATH:-}"      # makes `python -m exdrkg_rerun...` work without installing
SPLIT="${SPLIT:-standard}"; STRATEGY="${STRATEGY:-original}"   # original = the split files exactly as used in the manuscript (11,615 test pairs, as the plan requires); degree_matched | random = rebuilt negatives (leak-free, 100:1)
if [ "$STRATEGY" = original ]; then S_NAME="$SPLIT"; else S_NAME="${SPLIT}_${STRATEGY}"; fi
GPU="${GPU:-0}"; export CUDA_VISIBLE_DEVICES="$GPU"; export PYTHONHASHSEED=42
BACKBONES=("BioMistral/BioMistral-7B" "meta-llama/Llama-3.1-8B-Instruct" "epfl-llm/meditron-7b")
API_MODEL="${API_MODEL:-gpt-4o-2024-08-06}"                 # NEW recent model (Step 11)
LEGACY_API_MODEL="${LEGACY_API_MODEL:-gpt-4-turbo-2024-04-09}"   # the model of the submitted GPT-4+RAG baseline (Steps 2 and 4)
mkdir -p logs/rerun results/rerun

stage_splits() {
  # 1. clean negatives first (disjoint across splits, no duplicates), then freeze everything on the NEW split
  if [ "$STRATEGY" != original ]; then
    python -m exdrkg_rerun.data.rebuild_splits --root . --split "$SPLIT" --strategy "$STRATEGY" --seed 42 2>&1 | tee logs/rerun/rebuild_splits.log
  else echo "STRATEGY=original: using the split files as shipped (sensitivity run with rebuilt negatives: STRATEGY=degree_matched)"; fi
  # 2. masked knowledge graph + hashes + environment record (Table R-K)
  python -m exdrkg_rerun.data.masked_kg --root . --split "$SPLIT" 2>&1 | tee logs/rerun/masked_kg.log
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/rerun/gpu_info.csv || true
  pip freeze > results/rerun/pip_freeze.txt
}

stage_graph_baselines() {
  # graph-only baselines and the shortest-path sanity check, on the ORIGINAL and the REBUILT splits
  PREV=""
  for S in "$SPLIT" "${S_NAME}" "${SPLIT}_random"; do
    [ -f "data/splits/$S/train.csv" ] || continue; [ "$S" = "$PREV" ] && continue; PREV="$S"
    python -m exdrkg_rerun.baselines.graph_baselines --root . --split "$S" --boot 1000 2>&1 | tee "logs/rerun/graph_baselines_${S}.log"
  done
}

stage_depth() {
  # one job per depth so a failure never costs the others. L=6 is NOT retrained: it uses the existing V1 checkpoint.
  # Set V1_CKPT, V1_STAGE1_H, V1_PEAK_GB (from the original Stage-1 log) before running.
  for L in 2 4 8 10; do
    PLM=""; [ "$L" = "10" ] && PLM="--per-layer-mad"
    python -m exdrkg_rerun.models.depth_ablation --root . --split "${S_NAME}" --layers "$L" --epochs 100 $PLM 2>&1 | tee "logs/rerun/depth_L${L}.log"
  done
  python -m exdrkg_rerun.models.depth_ablation --root . --split "${S_NAME}" --layers 6 --checkpoint "${V1_CKPT:?set V1_CKPT}" \
    --stage1-hours "${V1_STAGE1_H:?}" --peak-gpu-gb "${V1_PEAK_GB:?}" --per-layer-mad 2>&1 | tee logs/rerun/depth_L6.log
  python -m exdrkg_rerun.evaluation.depth_summary --root . --split "${S_NAME}" 2>&1 | tee logs/rerun/depth_summary.log
}

stage_backbones() {
  # Identical Stage-2 (V4, no RLHF) for the reference and the two alternatives. Set exact names (and optionally revisions) below.
  GEN="${BACKBONE_GENERAL:-meta-llama/Llama-3.1-8B-Instruct}"; BIO="${BACKBONE_BIOMED:-epfl-llm/meditron-7b}"; REF="BioMistral/BioMistral-7B"; RUNS=()
  HB="results/rerun/hallucination_backbones"; mkdir -p "$HB"; cp results/rerun/hallucination/eval_sets.json results/rerun/hallucination/evidence.jsonl "$HB"/; rm -f "$HB/explanations.jsonl"
  EXT=""; if [ -f results/rerun/ablation/scores/V4.csv ]; then EXT="--external-ref-scores results/rerun/ablation/scores/V4.csv ${V4_STAGE2_H:+--external-ref-hours $V4_STAGE2_H}"; MODELS=("$GEN" "$BIO"); else MODELS=("$REF" "$GEN" "$BIO"); fi   # plan rule: BioMistral row = V4 of R-B
  for M in "${MODELS[@]}"; do
    TAG=$(echo "$M" | tr '/' '_'); OUT="models/stage2_${TAG}_V4"; NAME="ExDR-KG-${TAG}-V4"
    [ -f "$OUT/run.json" ] || python -m exdrkg_rerun.models.stage2_finetune --model "$M" --data data/llm/pairs.jsonl --out "$OUT" --epochs 10 --lr 2e-5 --lambda-faith 0.5 --seed 42 2>&1 | tee "logs/rerun/stage2_${TAG}_V4.log"
    python -m exdrkg_rerun.models.score_stage2 --adapter "$OUT" --name "$NAME" 2>&1 | tee "logs/rerun/score_${TAG}_V4.log"
    python -m exdrkg_rerun.explainability.generate_explanations --kind lora --sets "$HB" --model-name "$NAME" --hf-model "$M" --adapter "$OUT"
    RUNS+=("${NAME}=${OUT}")
  done
  python -m exdrkg_rerun.explainability.hallucination_eval run --sets "$HB" --vocab data/kg_entity_names.json --models "${RUNS[@]%%=*}" 2>&1 | tee logs/rerun/hallucination_backbones.log
  python -m exdrkg_rerun.evaluation.backbone_summary --root . --hallu "$HB/R-D_detail.json" $EXT --runs "${RUNS[@]}" 2>&1 | tee logs/rerun/step10.log
}

stage_stage2() {
  # Stage 2 with the differentiable score head. V3 (no faithfulness) and V4 (with faithfulness) per backbone.
  for M in "${BACKBONES[@]}"; do
    TAG=$(echo "$M" | tr '/' '_')
    python -m exdrkg_rerun.models.stage2_finetune --model "$M" --data data/llm/pairs.jsonl --out "models/stage2_${TAG}_V4" 2>&1 | tee "logs/rerun/stage2_${TAG}_V4.log"
  done
  python -m exdrkg_rerun.models.stage2_finetune --model "${BACKBONES[0]}" --data data/llm/pairs.jsonl --out "models/stage2_BioMistral_V3" --no-faith 2>&1 | tee logs/rerun/stage2_BioMistral_V3.log
}

stage_llm_baselines() {
  # Step 11: recent model + text RAG, and recent model + ExDR-KG paths (zero-shot). Same model string, same pairs, temperature 0.
  # Cost limit: export SAMPLE_NEG=<N> to score all positives + the same fixed N negatives for every model (sample file is reused).
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
  SAMP=""; [ -n "${SAMPLE_NEG:-}" ] && SAMP="--sample-negatives ${SAMPLE_NEG} --sample-file results/rerun/llm/sample_ids.csv"
  for CTX in rag paths; do
    NM="${API_LABEL:-GPT-4o}+RAG"; [ "$CTX" = paths ] && NM="${API_LABEL:-GPT-4o}+ExDR-KG paths (zero-shot)"
    python -m exdrkg_rerun.baselines.llm_api --model "$API_MODEL" --context "$CTX" --data data/llm/test_pairs.jsonl --out results/rerun/llm --score-name "$NM" $SAMP ${PROMPT_FILE:+--prompt-file $PROMPT_FILE} 2>&1 | tee "logs/rerun/api_${CTX}.log"
  done
  TAG=$(echo "$API_MODEL" | tr '/' '_'); SM=""; [ -n "${SAMPLE_NEG:-}" ] && SM="--sample results/rerun/llm/sample_ids.csv"
  # hallucination on H1 for the two new API rows (plan R-H last column): explanations on the H1/H2 pairs, same NER check as Step 4
  HL="results/rerun/hallucination_llm"; mkdir -p "$HL"; cp results/rerun/hallucination/eval_sets.json results/rerun/hallucination/evidence.jsonl results/rerun/hallucination/H1_pairs.txt results/rerun/hallucination/H2_pairs.txt "$HL"/
  grep '"model": "GPT-4+RAG"' results/rerun/hallucination/explanations.jsonl > "$HL/explanations.jsonl" || : > "$HL/explanations.jsonl"
  for CTX in rag paths; do
    NM="${API_LABEL:-GPT-4o}+RAG"; [ "$CTX" = paths ] && NM="${API_LABEL:-GPT-4o}+ExDR-KG paths (zero-shot)"
    python -m exdrkg_rerun.baselines.llm_api --model "$API_MODEL" --context "$CTX" --data data/llm/hallu_pairs.jsonl --out results/rerun/llm_hallu 2>&1 | tee "logs/rerun/api_hallu_${CTX}.log"
    python -m exdrkg_rerun.explainability.generate_explanations --kind api_convert --sets "$HL" --model-name "$NM" --api-expl "results/rerun/llm_hallu/expl_${TAG}_${CTX}.jsonl"
  done
  python -m exdrkg_rerun.explainability.hallucination_eval run --sets "$HL" --vocab data/kg_entity_names.json --models "GPT-4+RAG" "${API_LABEL:-GPT-4o}+RAG" "${API_LABEL:-GPT-4o}+ExDR-KG paths (zero-shot)" 2>&1 | tee logs/rerun/hallucination_llm.log
  python -m exdrkg_rerun.evaluation.llm_baseline_summary --root . --models "GPT-4+RAG" "${API_LABEL:-GPT-4o}+RAG" "${API_LABEL:-GPT-4o}+ExDR-KG paths (zero-shot)" $SM --hallu "$HL/R-D_detail.json" \
    ${SAMPLE_NEG:+--scores results/rerun/main/sample_scores} --runs "results/rerun/llm/run_${TAG}_rag.json" "results/rerun/llm/run_${TAG}_paths.json" --rag-config data/rag_config.json 2>&1 | tee logs/rerun/step11.log
}

stage_main_eval() {
  MASKED="data/processed/masked_${SPLIT}.pt"; S="$S_NAME"
  for M in rgcn compgcn han metapath2vec; do
    python -m exdrkg_rerun.baselines.gnn_baselines --root . --split "$S" --masked "$MASKED" --model "$M" 2>&1 | tee "logs/rerun/gnn_${M}.log"
  done
  python -m exdrkg_rerun.baselines.txgnn --root . --split "$S" --id-map data/txgnn_id_map.csv --txgnn-data data/raw/txgnn/data 2>&1 | tee logs/rerun/txgnn.log
  python -m exdrkg_rerun.baselines.biogpt --data data/llm/test_pairs.jsonl 2>&1 | tee logs/rerun/biogpt.log
  python -m exdrkg_rerun.baselines.llm_api --model "$LEGACY_API_MODEL" --context rag --data data/llm/test_pairs.jsonl --out results/rerun/llm --score-name "GPT-4+RAG" 2>&1 | tee logs/rerun/gpt4rag.log
  python -m exdrkg_rerun.models.score_exdrkg --root . --split "$S" --masked "$MASKED" --checkpoint models/exdr_kg_v5.pt --tag ExDR-KG 2>&1 | tee logs/rerun/score_exdrkg.log
  python -m exdrkg_rerun.evaluation.main_evaluation --root . --split "$S" --family-map data/disease_family.csv --ref ExDR-KG 2>&1 | tee logs/rerun/step2.log
}

stage_ablation() {
  MASKED="data/processed/masked_${SPLIT}.pt"; S="$S_NAME"
  [ -f rerun/configs/ablation_manifest.yaml ] || { echo "copy rerun/configs/ablation_manifest.example.yaml to rerun/configs/ablation_manifest.yaml and set the checkpoint paths"; exit 1; }
  python -m exdrkg_rerun.models.score_ablation_variants --root . --split "$S" --masked "$MASKED" --manifest rerun/configs/ablation_manifest.yaml 2>&1 | tee logs/rerun/score_ablation.log
  python -m exdrkg_rerun.evaluation.ablation_evaluation --root . --split "$S" --family-map data/disease_family.csv 2>&1 | tee logs/rerun/step3.log
  python -m exdrkg_rerun.models.check_ppo_lora_sharing --v4 models/ablation/v4 --v5 models/ablation/v5 --out results/rerun/ablation/ppo_lora_check.json 2>&1 | tee logs/rerun/ppo_lora.log
}

stage_hallucination() {
  MASKED="data/processed/masked_${SPLIT}.pt"; S="$S_NAME"; H="results/rerun/hallucination"
  # H1 = the 200 expert-rated pairs (put their IDs in data/expert_pairs_200.txt, one drug_id|disease_id per line)
  python -m exdrkg_rerun.explainability.build_eval_sets --root . --split "$S" --h1-file data/expert_pairs_200.txt --h2-mode test_sample --n-h2 500 2>&1 | tee logs/rerun/eval_sets.log
  python -m exdrkg_rerun.explainability.build_evidence --root . --split "$S" --masked "$MASKED" --encoder models/hkg_encoder.pt --retriever models/retriever.pt --names data/entity_names.json 2>&1 | tee logs/rerun/evidence.log
  rm -f "$H/explanations.jsonl"
  python -m exdrkg_rerun.explainability.generate_explanations --kind biogpt --model-name BioGPT --hf-model microsoft/biogpt
  python -m exdrkg_rerun.explainability.generate_explanations --kind api_input --passages data/llm/rag_passages.jsonl --out-api data/llm/hallu_pairs.jsonl
  python -m exdrkg_rerun.baselines.llm_api --model "$LEGACY_API_MODEL" --context rag --data data/llm/hallu_pairs.jsonl --out results/rerun/llm_hallu
  python -m exdrkg_rerun.explainability.generate_explanations --kind api_convert --model-name GPT-4+RAG --api-expl "results/rerun/llm_hallu/expl_${LEGACY_API_MODEL}_rag.jsonl"
  python -m exdrkg_rerun.explainability.generate_explanations --kind txgnn_template --model-name "TxGNN+Template" --txgnn-paths data/txgnn_top_paths.jsonl
  for V in V3 V4 V5; do
    python -m exdrkg_rerun.explainability.generate_explanations --kind lora --model-name "$V" --hf-model BioMistral/BioMistral-7B --adapter "models/stage2_BioMistral_${V}"
  done
  python -m exdrkg_rerun.explainability.hallucination_eval run --vocab data/kg_entity_names.json --models BioGPT GPT-4+RAG TxGNN+Template V3 V4 V5 2>&1 | tee logs/rerun/hallucination.log
}

stage_hkg_stats() {
  MASKED="data/processed/masked_${SPLIT}.pt"
  python -m exdrkg_rerun.graph_analysis.hkg_statistics --root . --masked "$MASKED" --exact-diameter 2>&1 | tee logs/rerun/step5.log
  python -m exdrkg_rerun.data.kg_harmonisation --root . --kg data/processed/unified_kg.pt --logs logs/build_kg.log 2>&1 | tee logs/rerun/harmonisation.log
}

stage_cases() {
  python -m exdrkg_rerun.explainability.case_studies --root . --scores results/rerun/main/scores/ExDR-KG_temporal.csv --test data/splits/temporal/test.csv \
    --train-kg data/processed/masked_temporal.pt --literature data/case_literature.csv --evidence results/rerun/cases/evidence.jsonl \
    --explanations results/rerun/cases/explanations.jsonl --list-candidates 2>&1 | tee logs/rerun/step7.log
}
stage_rlhf_audit() {
  python -m exdrkg_rerun.human_eval.rlhf_audit --root . --rlhf data/rlhf/preferences.csv --eval results/expert/ratings.csv --eval-pairs data/expert_pairs_200.txt 2>&1 | tee logs/rerun/step8.log
}
stage_paths() {
  python -m exdrkg_rerun.graph_analysis.path_availability --root . --jobs "${SPLIT}:test" zero_shot:test temporal:test --cap 1000000 2>&1 | tee logs/rerun/step6.log
}

stage_facts() {
  python -m exdrkg_rerun.reporting.facts_sheet --root . --split "$SPLIT" 2>&1 | tee logs/rerun/facts.log
}

stage_figures() {
  S="$S_NAME"
  python -m exdrkg_rerun.reporting.figures degree --root . --masked "data/processed/masked_${SPLIT}.pt" 2>&1 | tee logs/rerun/fig_degree.log
  python -m exdrkg_rerun.reporting.figures fig13 --root . --include-failures 2>&1 | tee logs/rerun/fig13.log
  echo "fig5 and fig11 need extra inputs; run them by hand (see rerun/README.md, section Figures)"
}

stage_language() {
  : "${MANUSCRIPT:?set MANUSCRIPT to the .docx path}"
  python -m exdrkg_rerun.reporting.language_audit audit --docx "$MANUSCRIPT" 2>&1 | tee logs/rerun/language_audit.log
}

usage() { sed -n '/^# stage /,/^# (Step 12/p' "$0" | sed 's/^# //'; }
case "${1:-help}" in
  splits) stage_splits ;; graph-baselines) stage_graph_baselines ;; main-eval) stage_main_eval ;; ablation) stage_ablation ;; hallucination) stage_hallucination ;;
  hkg-stats) stage_hkg_stats ;; paths) stage_paths ;; cases) stage_cases ;; rlhf-audit) stage_rlhf_audit ;; depth) stage_depth ;; backbones) stage_backbones ;;
  llm-baselines) stage_llm_baselines ;; stage2) stage_stage2 ;; facts) stage_facts ;; figures) stage_figures ;; language) stage_language ;;
  help|-h|--help) usage ;; *) echo "unknown stage '${1}'"; usage; exit 1 ;;
esac
