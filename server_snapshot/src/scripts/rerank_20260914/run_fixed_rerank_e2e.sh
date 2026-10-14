#!/usr/bin/env bash
# End-to-end verification of the FIXED front-end reranker (apply_rerank_v2.py:
# exact-weighted primary key + length-normalised forced CTC + no-shorten floor),
# at the relaxed admission, with NO trust anchor, on AISHELL dev / THCHS-30 / ST-CMDS.
#
# Waits for the two jobs already on the GPU before touching it (a previous
# concurrent run cost 3.7 h of GPU to a mis-read decode).
set -uo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_AI="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
LOG="$R/fixed_rerank_e2e.log"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
MAN_A="$WS/src/logs/hotword_lora_9b_20260908/dual_ablation_20260909/audio_manifest.jsonl"

echo "[$(date -Is)] waiting for GPU (STCMDS_SAFE_DONE + SAMECODE_DONE)" >> "$LOG"
for i in $(seq 1 600); do
  if [[ -f "$R/STCMDS_SAFE_DONE" && -f "$R/SAMECODE_DONE" ]]; then break; fi
  sleep 60
done
echo "[$(date -Is)] GPU free, start" >> "$LOG"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

# 0) ST-CMDS relaxed-pool forced-CTC scores were truncated by an earlier crash (1247/5130)
if [[ "$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl" 2>/dev/null || echo 0)" -lt 5130 ]]; then
  rm -f "$R/stcmds_relax_ctc_scores.jsonl"
  echo "[$(date -Is)] rescoring ST-CMDS relaxed pool" >> "$LOG"
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/stcmds_relax_ctc_input.jsonl" --manifest "$R/stcmds_manifest.jsonl" \
    --output "$R/stcmds_relax_ctc_scores.jsonl" --max-nbest 24 --progress-every 500 >> "$LOG" 2>&1
fi
echo "[$(date -Is)] stcmds relax ctc rows=$(wc -l < "$R/stcmds_relax_ctc_scores.jsonl")" >> "$LOG"

# 1) fixed rerank on all three relaxed pools
{
  "$PY" "$WS/.dsh_checks/apply_rerank_v2.py" \
    --evidence "$R/dev_relax.jsonl" --ctc-scores "$R/relax_ctc_scores.jsonl" \
    --uttid "$A/uttid" --aligned "$A/aligned.txt" \
    --label "AISHELL dev / relaxed + FIXED rerank" --out "$R/dev_relax_fixed.jsonl"
  "$PY" "$WS/.dsh_checks/apply_rerank_v2.py" \
    --evidence "$R/thchs_relax.jsonl" --ctc-scores "$R/thchs_ctc_scores.jsonl" \
    --uttid "$T/uttid" --aligned "$T/aligned.txt" \
    --label "THCHS-30 / relaxed + FIXED rerank" --out "$R/thchs_relax_fixed.jsonl"
  "$PY" "$WS/.dsh_checks/apply_rerank_v2.py" \
    --evidence "$R/stcmds_relax.jsonl" --ctc-scores "$R/stcmds_relax_ctc_scores.jsonl" \
    --uttid "$S/uttid" --aligned "$S/aligned.txt" \
    --label "ST-CMDS / relaxed + FIXED rerank" --out "$R/stcmds_relax_fixed.jsonl"
} 2>&1 | tee -a "$LOG"

run_arm() {  # evidence, tag, adapter, aligned, uttid
  local ev="$1" tag="$2" ada="$3" alg="$4" utt="$5"
  local msg="$R/${tag}.messages.jsonl" pred="$R/${tag}.predictions.jsonl"
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
      --model-name-or-path "$MODEL" --adapter-path "$ada" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
    mv -f "$pred.inprogress" "$pred"
  fi
  echo "[$(date -Is)] $tag ready" >> "$LOG"
}

# 2) AISHELL dev first (tuning set, ~10 min)
run_arm "$R/dev_relax_fixed.jsonl" armG1 "$ADA_AI" "$A/aligned.txt" "$A/uttid"
# 3) ST-CMDS held-out (the failing transfer case, ~1.7 h)
run_arm "$R/stcmds_relax_fixed.jsonl" e2eSTCMDSFIX "$ADA_ST" "$S/aligned.txt" "$S/uttid"
# 4) THCHS-30 (~45 min) for a consistent table
run_arm "$R/thchs_relax_fixed.jsonl" e2eTHCHSFIX "$ADA_AI" "$T/aligned.txt" "$T/uttid"

# 5) metrics
{
  echo "================ FIXED-RERANK END-TO-END ================"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$R/armG1.predictions.jsonl" --label "AISHELL fixed-rerank e2e" \
    --aligned "$A/aligned.txt" --uttid-file "$A/uttid"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$R/e2eSTCMDSFIX.predictions.jsonl" --label "ST-CMDS fixed-rerank e2e" \
    --aligned "$S/aligned.txt" --uttid-file "$S/uttid"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$R/e2eTHCHSFIX.predictions.jsonl" --label "THCHS-30 fixed-rerank e2e" \
    --aligned "$T/aligned.txt" --uttid-file "$T/uttid"
} >> "$LOG" 2>&1

touch "$R/FIXED_RERANK_E2E_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
