#!/usr/bin/env bash
# Round 3 acceptance: does the raised-ceiling front-end help (or at least not hurt) COVO?
#   E1 = relaxed evidence, no trust anchor
#   E2 = relaxed evidence + protect/forced-CTC rerank, no trust anchor
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
LOG="$R/round3.log"
MAN="$WS/src/logs/hotword_lora_9b_20260908/dual_ablation_20260909/audio_manifest.jsonl"

# 1) forced-CTC scores for the relaxed pool, then rerank
"$PY" "$R/build_pool_ctc_input.py" relax >> "$LOG" 2>&1 || true
if [[ ! -s "$R/relax_ctc_scores.jsonl" ]]; then
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/relax_ctc_input.jsonl" --manifest "$MAN" \
    --output "$R/relax_ctc_scores.jsonl" --max-nbest 24 --progress-every 200 >> "$LOG" 2>&1
fi
"$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
  --evidence "$R/dev_relax.jsonl" --ctc-scores "$R/relax_ctc_scores.jsonl" \
  --uttid "$A/uttid" --aligned "$A/aligned.txt" \
  --label "relax front-end + protect/forced-CTC rerank" --out "$R/dev_relax_reranked.jsonl" >> "$LOG" 2>&1

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

run_arm() {
  local ev="$1" tag="$2"
  local msg="$R/arm${tag}.messages.jsonl" pred="$R/arm${tag}.predictions.jsonl"
  if [[ ! -s "$msg" ]]; then
    cd "$WS"
    "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$ev" --output "$msg.inprogress" \
      "${COMMON[@]}" >> "$LOG" 2>&1
    mv -f "$msg.inprogress" "$msg"
  fi
  if [[ ! -s "$pred" ]]; then
    cd "$COVO"
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
      --input "$msg" --output "$pred.inprogress" \
      --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
    mv -f "$pred.inprogress" "$pred"
  fi
  echo "[$(date -Is)] $tag ready" >> "$LOG"
}

run_arm "$R/dev_relax.jsonl" F1
run_arm "$R/dev_relax_reranked.jsonl" F2

touch "$R/ROUND3_DONE"
echo "[$(date -Is)] ROUND3 DONE" >> "$LOG"
