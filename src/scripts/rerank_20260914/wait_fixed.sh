#!/usr/bin/env bash
# Wait for the fixed-rerank end-to-end queue and print every result it produced.
R=/root/autodl-tmp/.dsh_checks/rerank
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
A=/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev
T=/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test
PY=/root/autodl-tmp/great/bin/python
cd /root/autodl-tmp

for i in $(seq 1 900); do
  [ -f "$R/FIXED_RERANK_E2E_DONE" ] && break
  sleep 60
done

echo "############ fixed-rerank front-end (relaxed admission) ############"
cat "$R/fixed_rerank_e2e.log" | grep -A3 '^# ' | head -40
echo
echo "############ ST-CMDS relaxed + FIXED rerank, end-to-end ############"
[ -s "$R/e2eSTCMDSFIX.predictions.jsonl" ] && "$PY" .dsh_checks/restore_eval.py \
  --records "$R/e2eSTCMDSFIX.predictions.jsonl" --label "ST-CMDS fixed-rerank e2e" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid"
echo
echo "############ AISHELL relaxed + FIXED rerank, end-to-end ############"
[ -s "$R/armG1.predictions.jsonl" ] && "$PY" .dsh_checks/restore_eval.py \
  --records "$R/armG1.predictions.jsonl" --label "AISHELL fixed-rerank e2e" \
  --aligned "$A/aligned.txt" --uttid-file "$A/uttid"
echo
echo "############ THCHS-30 relaxed + FIXED rerank, end-to-end ############"
[ -s "$R/e2eTHCHSFIX.predictions.jsonl" ] && "$PY" .dsh_checks/restore_eval.py \
  --records "$R/e2eTHCHSFIX.predictions.jsonl" --label "THCHS-30 fixed-rerank e2e" \
  --aligned "$T/aligned.txt" --uttid-file "$T/uttid"
echo
echo "############ tail of the queue log ############"
tail -6 "$R/fixed_rerank_e2e.log"
