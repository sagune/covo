#!/usr/bin/env bash
# Wait for the ST-CMDS original-admission fixed-rerank e2e and print its table.
R=/root/autodl-tmp/.dsh_checks/rerank
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
PY=/root/autodl-tmp/great/bin/python
cd /root/autodl-tmp
for i in $(seq 1 900); do
  [ -f "$R/STCMDS_ORIG_FIXED_E2E_DONE" ] && break
  sleep 60
done
echo "############ ST-CMDS original admission + FIXED rerank, end-to-end ############"
tail -12 "$R/stcmds_orig_fixed_e2e.log"
echo
echo "############ explicit re-eval ############"
[ -s "$R/e2eSTCMDSORIGFIX.predictions.jsonl" ] && "$PY" .dsh_checks/restore_eval.py \
  --records "$R/e2eSTCMDSORIGFIX.predictions.jsonl" --label "ST-CMDS orig+fixed e2e" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid"
