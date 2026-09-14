#!/usr/bin/env bash
# Queue v2.  The learned ranker (idea A) turned out to be the strongest front-end
# result of the session, so its downstream test now outranks the breadth split and
# the same-code baseline decode, which move to the back.
#
#   1  THCHS-30 relaxed + FIXED rescorer            (~45 min)
#   2  ST-CMDS  relaxed + FIXED rescorer            (~1.8 h, incl. CTC rescoring)
#   3  ST-CMDS  + LEARNED ranker (trained on AISHELL dev)   (~1.7 h)   <- headline
#   4  THCHS-30 + LEARNED ranker                    (~45 min)
#   5  breadth-only admission split                 (~4 h, self-gated)
#   6  same-code ST-CMDS baseline decode            (~2 h)
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
LOG="$R/queue2_run.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "================ QUEUE v2 START ================"
for i in $(seq 1 15); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"
  sleep 120
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

# --- 1: THCHS-30 relaxed + FIXED (was interrupted, restarts from scratch) -----
say "---------- step 1: THCHS-30 relaxed + FIXED rescorer ----------"
infer  "$R/e2eTHCHSFIX.messages.jsonl" "$R/e2eTHCHSFIX.predictions.jsonl" "$ADA_AI" "THCHS relax+fixed"
evaluate "$R/e2eTHCHSFIX.predictions.jsonl" "THCHS-30 relax+fixed e2e" "$T/aligned.txt" "$T/uttid"
touch "$R/FIXED_RERANK_E2E_DONE"

# --- 2: ST-CMDS relaxed + FIXED ----------------------------------------------
say "---------- step 2: ST-CMDS relaxed + FIXED rescorer ----------"
n=$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl" 2>/dev/null || echo 0)
if [[ "$n" -lt 5130 ]]; then
  say "rescoring ST-CMDS relaxed pool (had $n)"
  rm -f "$R/stcmds_relax_ctc_scores.jsonl"
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/stcmds_relax_ctc_input.jsonl" --manifest "$R/stcmds_manifest.jsonl" \
    --output "$R/stcmds_relax_ctc_scores.jsonl" --max-nbest 24 --progress-every 500 >> "$LOG" 2>&1
  say "rescore done rows=$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl" 2>/dev/null || echo 0)"
fi
if [[ ! -s "$R/stcmds_relax_fixed.jsonl" ]]; then
  "$PY" "$WS/.dsh_checks/apply_rerank_v2.py" --evidence "$R/stcmds_relax.jsonl" \
    --ctc-scores "$R/stcmds_relax_ctc_scores.jsonl" --uttid "$S/uttid" --aligned "$S/aligned.txt" \
    --label "ST-CMDS / relaxed + FIXED rerank" --out "$R/stcmds_relax_fixed.jsonl" >> "$LOG" 2>&1
fi
if [[ ! -s "$R/e2eSTCMDSFIX.messages.jsonl" ]]; then
  cd "$WS"
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_relax_fixed.jsonl" \
    --output "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin \
    --prompt-mode selector --protect-supported-hotwords --clean-nbest --include-consensus-spans \
    --include-hotword-evidence >> "$LOG" 2>&1
  [[ -s "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" ]] && mv -f "$R/e2eSTCMDSFIX.messages.jsonl.inprogress" "$R/e2eSTCMDSFIX.messages.jsonl"
fi
infer  "$R/e2eSTCMDSFIX.messages.jsonl" "$R/e2eSTCMDSFIX.predictions.jsonl" "$ADA_ST" "ST-CMDS relax+fixed"
evaluate "$R/e2eSTCMDSFIX.predictions.jsonl" "ST-CMDS relax+fixed e2e" "$S/aligned.txt" "$S/uttid"
touch "$R/STCMDS_ORIG_FIXED_E2E_DONE"

# --- 3: learned ranker on ST-CMDS (trained on AISHELL dev) --------------------
say "---------- step 3: ST-CMDS + LEARNED ranker ----------"
infer  "$R/e2eSTCMDSLEARN.messages.jsonl" "$R/e2eSTCMDSLEARN.predictions.jsonl" "$ADA_ST" "ST-CMDS learned"
evaluate "$R/e2eSTCMDSLEARN.predictions.jsonl" "ST-CMDS learned-ranker e2e" "$S/aligned.txt" "$S/uttid"
touch "$R/STCMDS_LEARNED_E2E_DONE"

# --- 4: learned ranker on THCHS-30 ------------------------------------------
say "---------- step 4: THCHS-30 + LEARNED ranker ----------"
infer  "$R/e2eTHCHSLEARN.messages.jsonl" "$R/e2eTHCHSLEARN.predictions.jsonl" "$ADA_AI" "THCHS learned"
evaluate "$R/e2eTHCHSLEARN.predictions.jsonl" "THCHS-30 learned-ranker e2e" "$T/aligned.txt" "$T/uttid"
touch "$R/THCHS_LEARNED_E2E_DONE"

# --- 5: breadth split --------------------------------------------------------
say "---------- step 5: breadth-only admission split ----------"
if [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]]; then
  say "breadth already settled, skipping"
else
  nohup bash "$WS/.dsh_checks/run_stcmds_breadth.sh" >> "$LOG" 2>&1 &
  for i in $(seq 1 480); do
    [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]] && break
    sleep 60
  done
  say "breadth finished"
fi

# --- 6: same-code baseline --------------------------------------------------
say "---------- step 6: same-code ST-CMDS baseline ----------"
bash "$WS/.dsh_checks/run_samecode_baselines.sh" >> "$LOG" 2>&1
say "samecode done: $([[ -f "$R/SAMECODE_DONE" ]] && echo yes || echo no)"

touch "$R/QUEUE2_DONE"
say "================ QUEUE v2 DONE ================"
