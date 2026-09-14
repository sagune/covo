#!/usr/bin/env bash
# Rerank-verification queue, ordered by information per GPU-hour.
#
#   1  ST-CMDS original admission + FIXED rescorer, end to end   (~1.7 h)  <- decisive
#   2  AISHELL dev   relaxed + FIXED rescorer, end to end        (~10 min)
#   3  THCHS-30      relaxed + FIXED rescorer, end to end        (~45 min)
#   4  ST-CMDS relaxed + FIXED rescorer, end to end              (~1.8 h, incl. 6 min rescoring)
#   5  breadth-only admission split on ST-CMDS                   (~4 h, self-gated)
#   6  same-code ST-CMDS baseline decode + pool comparison       (~2 h)
#
# Step 1 is first because its two reference points already exist: the baseline
# (5.0300%/93.48%/destroyed 0) and the shipped rescorer at the same admission
# (6.2657%/92.90%/0).  One run then answers "did the fix remove that +1.2357 pp".
#
# All steps are individually guarded and idempotent; a failure logs and moves on.
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
LOG="$R/queue_run.log"

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "================ QUEUE START ================"

# --- wait for the GPU to be free (cap 30 min) --------------------------------
for i in $(seq 1 15); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy (pids: $busy), waiting"
  sleep 120
done
say "gpu state: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ' ')(empty means free)"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

infer() {  # messages, predictions, adapter, label
  local msg="$1" pred="$2" ada="$3" label="$4"
  if [[ -s "$pred" ]]; then say "SKIP infer ($label): product exists"; return 0; fi
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  if [[ -s "$pred.inprogress" ]]; then
    mv -f "$pred.inprogress" "$pred"
    say "infer DONE ($label) rows=$(wc -l < "$pred")"
  else
    say "infer FAILED ($label): no output"
  fi
  cd "$WS"
}

evaluate() {  # predictions, label, aligned, uttid
  [[ -s "$1" ]] || { say "eval SKIP ($2): no predictions"; return 0; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" \
    --aligned "$3" --uttid-file "$4" >> "$LOG" 2>&1
  say "eval done ($2)"
}

# ============================ step 1 =========================================
say "---------- step 1: ST-CMDS original admission + FIXED rescorer ----------"
infer  "$R/e2eSTCMDSORIGFIX.messages.jsonl" "$R/e2eSTCMDSORIGFIX.predictions.jsonl" "$ADA_ST" "ST-CMDS orig+fixed"
evaluate "$R/e2eSTCMDSORIGFIX.predictions.jsonl" "ST-CMDS orig+fixed e2e" "$S/aligned.txt" "$S/uttid"
touch "$R/STCMDS_ORIG_FIXED_E2E_DONE"

# ============================ step 2 =========================================
say "---------- step 2: AISHELL dev relaxed + FIXED rescorer ----------"
infer  "$R/armG1.messages.jsonl" "$R/armG1.predictions.jsonl" "$ADA_AI" "AISHELL relax+fixed"
evaluate "$R/armG1.predictions.jsonl" "AISHELL relax+fixed e2e" "$A/aligned.txt" "$A/uttid"

# ============================ step 3 =========================================
say "---------- step 3: THCHS-30 relaxed + FIXED rescorer ----------"
infer  "$R/e2eTHCHSFIX.messages.jsonl" "$R/e2eTHCHSFIX.predictions.jsonl" "$ADA_AI" "THCHS relax+fixed"
evaluate "$R/e2eTHCHSFIX.predictions.jsonl" "THCHS-30 relax+fixed e2e" "$T/aligned.txt" "$T/uttid"
touch "$R/FIXED_RERANK_E2E_DONE"

# ============================ step 4 =========================================
say "---------- step 4: ST-CMDS relaxed + FIXED rescorer ----------"
n=$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl" 2>/dev/null || echo 0)
if [[ "$n" -lt 5130 ]]; then
  say "rescoring ST-CMDS relaxed pool (had $n rows)"
  rm -f "$R/stcmds_relax_ctc_scores.jsonl"
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/stcmds_relax_ctc_input.jsonl" --manifest "$R/stcmds_manifest.jsonl" \
    --output "$R/stcmds_relax_ctc_scores.jsonl" --max-nbest 24 --progress-every 500 >> "$LOG" 2>&1
  say "rescore done rows=$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl" 2>/dev/null || echo 0)"
fi
if [[ ! -s "$R/stcmds_relax_fixed.jsonl" ]]; then
  "$PY" "$WS/.dsh_checks/apply_rerank_v2.py" \
    --evidence "$R/stcmds_relax.jsonl" --ctc-scores "$R/stcmds_relax_ctc_scores.jsonl" \
    --uttid "$S/uttid" --aligned "$S/aligned.txt" \
    --label "ST-CMDS / relaxed + FIXED rerank" --out "$R/stcmds_relax_fixed.jsonl" >> "$LOG" 2>&1
fi
if [[ ! -s "$R/e2eSTCMDSFIX.messages.jsonl" ]]; then
  cd "$WS"
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_relax_fixed.jsonl" \
    --output "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" "${COMMON[@]}" >> "$LOG" 2>&1
  [[ -s "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" ]] && mv -f "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" "$R/e2eSTCMDSFIX.messages.jsonl"
fi
infer  "$R/e2eSTCMDSFIX.messages.jsonl" "$R/e2eSTCMDSFIX.predictions.jsonl" "$ADA_ST" "ST-CMDS relax+fixed"
evaluate "$R/e2eSTCMDSFIX.predictions.jsonl" "ST-CMDS relax+fixed e2e" "$S/aligned.txt" "$S/uttid"
touch "$R/STCMDS_ORIG_FIXED_E2E_DONE"

# ============================ step 5 =========================================
say "---------- step 5: breadth-only admission split ----------"
if [[ -f "$R/STCMDS_BREADTH_DONE" ]]; then
  say "breadth already done, skipping"
else
  nohup bash "$WS/.dsh_checks/run_stcmds_breadth.sh" >> "$LOG" 2>&1 &
  for i in $(seq 1 480); do
    [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]] && break
    sleep 60
  done
  say "breadth finished: DONE=$([[ -f "$R/STCMDS_BREADTH_DONE" ]] && echo yes || echo no) FAILED=$([[ -f "$R/STCMDS_BREADTH_FAILED" ]] && echo yes || echo no)"
fi

# ============================ step 6 =========================================
say "---------- step 6: same-code ST-CMDS baseline decode ----------"
bash "$WS/.dsh_checks/run_samecode_baselines.sh" >> "$LOG" 2>&1
say "samecode finished: SAMECODE_DONE=$([[ -f "$R/SAMECODE_DONE" ]] && echo yes || echo no)"

# ============================ summary ========================================
say "================ RESULTS ================"
{
  echo "########## fixed-rerank end-to-end ##########"
  grep -E "^# |input  |raw  |verifier  |restore\[deployable\]" "$LOG" | tail -60
} >> "$LOG" 2>&1
touch "$R/QUEUE_DONE"
say "================ QUEUE DONE ================"
