#!/usr/bin/env bash
# Queue v3.  The learned ranker's downstream already passed on ST-CMDS and
# THCHS-30 with the first (min-CER) variant; these are the improved variants that
# also hold recall in domain, plus the AISHELL arms that were never run.
#
#   1  AISHELL dev relaxed + learned v2   (~10 min)
#   2  AISHELL dev original + learned v2  (~10 min)
#   3  ST-CMDS + learned v2               (~1.7 h)
#   4  THCHS-30 + learned v2              (~45 min)
#   5  breadth-only admission split       (~4 h, self-gated)   [lower value now]
#   6  same-code ST-CMDS baseline decode  (~2 h)               [confound check]
set -uo pipefail
WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_AI="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/queue3_run.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
say "================ QUEUE v3 START ================"
for i in $(seq 1 20); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 120
done
infer() {
  local msg="$1" pred="$2" ada="$3" label="$4"
  if [[ -s "$pred" ]]; then say "SKIP infer ($label)"; return 0; fi
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  if [[ -s "$pred.inprogress" ]]; then mv -f "$pred.inprogress" "$pred"; say "infer DONE ($label) rows=$(wc -l < "$pred")"; else say "infer FAILED ($label)"; fi
  cd "$WS"
}
evaluate() {
  [[ -s "$1" ]] || { say "eval SKIP ($2)"; return 0; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" --aligned "$3" --uttid-file "$4" >> "$LOG" 2>&1
  say "eval done ($2)"
}
say "---------- step 1: AISHELL relaxed + learned v2 ----------"
infer  "$R/armG3.messages.jsonl" "$R/armG3.predictions.jsonl" "$ADA_AI" "AISHELL relax learned v2"
evaluate "$R/armG3.predictions.jsonl" "AISHELL relax + learned v2 e2e" "$A/aligned.txt" "$A/uttid"
say "---------- step 2: AISHELL original + learned v2 ----------"
infer  "$R/armG2.messages.jsonl" "$R/armG2.predictions.jsonl" "$ADA_AI" "AISHELL orig learned v2"
evaluate "$R/armG2.predictions.jsonl" "AISHELL orig + learned v2 e2e" "$A/aligned.txt" "$A/uttid"
touch "$R/AISHELL_LEARNED_E2E_DONE"
say "---------- step 3: ST-CMDS + learned v2 ----------"
infer  "$R/e2eSTCMDSLEARN2.messages.jsonl" "$R/e2eSTCMDSLEARN2.predictions.jsonl" "$ADA_ST" "ST-CMDS learned v2"
evaluate "$R/e2eSTCMDSLEARN2.predictions.jsonl" "ST-CMDS + learned v2 e2e" "$S/aligned.txt" "$S/uttid"
touch "$R/STCMDS_LEARNED2_E2E_DONE"
say "---------- step 4: THCHS-30 + learned v2 ----------"
infer  "$R/e2eTHCHSLEARN2.messages.jsonl" "$R/e2eTHCHSLEARN2.predictions.jsonl" "$ADA_AI" "THCHS learned v2"
evaluate "$R/e2eTHCHSLEARN2.predictions.jsonl" "THCHS-30 + learned v2 e2e" "$T/aligned.txt" "$T/uttid"
touch "$R/THCHS_LEARNED2_E2E_DONE"
say "---------- step 5: breadth split ----------"
if [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]]; then
  say "breadth settled, skipping"
else
  nohup bash "$WS/.dsh_checks/run_stcmds_breadth.sh" >> "$LOG" 2>&1 &
  for i in $(seq 1 480); do
    [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]] && break
    sleep 60
  done
  say "breadth finished"
fi
say "---------- step 6: same-code baseline ----------"
bash "$WS/.dsh_checks/run_samecode_baselines.sh" >> "$LOG" 2>&1
say "samecode done: $([[ -f "$R/SAMECODE_DONE" ]] && echo yes || echo no)"
touch "$R/QUEUE3_DONE"
say "================ QUEUE v3 DONE ================"
