#!/usr/bin/env bash
# Same-code baselines, to remove the code-drift confound discovered in round 19.
#   (a) THCHS-30 baseline end-to-end: thchs_base.jsonl with the SHIPPED bridge flags
#   (b) ST-CMDS held-out baseline decode (config defaults) once the queue drains
set -euo pipefail

WS=/root/autodl-tmp
SRC="$WS/src"
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_AI="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
LOG="$R/samecode.log"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
D="$WS/datasets/stcmds/cb_sensevoice_heldout"
ORIG_T="$COVO/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"

# ---------- (a) THCHS-30 shipped-config end-to-end baseline, current code ----------
if [[ ! -s "$R/thchs_base_predictions.jsonl" ]]; then
  echo "[$(date -Is)] THCHS shipped-config e2e baseline START" >> "$LOG"
  MSG="$R/thchs_base.messages.jsonl"
  cd "$WS"
  if [[ ! -s "$MSG" ]]; then
    "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare \
      --input "$R/thchs_base.jsonl" --output "$MSG.inprogress" \
      --max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6 \
      --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin \
      --prompt-mode selector --protect-supported-hotwords --clean-nbest \
      --include-consensus-spans --include-hotword-evidence --prefer-same-length \
      --trust-asr-top1 --preserve-anchor-digits >> "$LOG" 2>&1
    mv -f "$MSG.inprogress" "$MSG"
  fi
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$MSG" --output "$R/thchs_base_predictions.jsonl.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADA_AI" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  mv -f "$R/thchs_base_predictions.jsonl.inprogress" "$R/thchs_base_predictions.jsonl"
  echo "[$(date -Is)] THCHS baseline e2e DONE" >> "$LOG"
fi
echo "=== THCHS-30 same-code baseline vs new config (end-to-end) ===" >> "$LOG"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$R/thchs_base_predictions.jsonl" \
  --label "THCHS same-code baseline (shipped config)" --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$LOG" 2>&1

# ---------- (b) ST-CMDS held-out baseline decode, after the queue drains ----------
for i in $(seq 1 480); do
  [ -f "$R/STCMDS_RR_DONE" ] && break
  pgrep -f "run_CLI.py test" >/dev/null || { [ -f "$R/STCMDS_DONE" ] && break; }
  sleep 120
done
while pgrep -f "run_CLI.py test" >/dev/null; do sleep 60; done
echo "[$(date -Is)] ST-CMDS base decode START" >> "$LOG"
if [[ ! -s "$R/stcmds_base.jsonl" ]]; then
  cd "$SRC"
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$R/stcmds_base.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-stcmds.yaml \
    --data.init_args.test_info.root="../datasets/stcmds/cb_sensevoice_heldout" \
    --data.init_args.num_workers=0 \
    --model.init_args.root="../datasets/stcmds/cb_sensevoice_heldout/hotword" \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$R/stcmds_base_oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$R/stcmds_base_oracle_summary.csv" >> "$LOG" 2>&1
fi
"$PY" "$WS/.dsh_checks/compare_pool_generic.py" \
  --uttid "$D/hotword/test/uttid" --aligned "$D/hotword/test/aligned.txt" \
  --label "ST-CMDS held-out: 07-29 evidence vs same-code base vs relax" \
  --evidence "original(07-29)=$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
  --evidence "base(current code)=$R/stcmds_base.jsonl" \
  --evidence "relax=$R/stcmds_relax.jsonl" >> "$LOG" 2>&1

touch "$R/SAMECODE_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
