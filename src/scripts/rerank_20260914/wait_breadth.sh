#!/usr/bin/env bash
# Wait for the breadth-only admission split on ST-CMDS and print everything.
R=/root/autodl-tmp/.dsh_checks/rerank
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
PY=/root/autodl-tmp/great/bin/python
cd /root/autodl-tmp
for i in $(seq 1 1200); do
  [ -f "$R/STCMDS_BREADTH_DONE" ] && break
  [ -f "$R/STCMDS_BREADTH_FAILED" ] && break
  sleep 60
done
echo "############ breadth-only relax front-end comparison ############"
awk '/breadth-only relax/,0' "$R/stcmds_breadth.log" | head -20
echo
echo "############ breadth-only + FIXED rerank ############"
cat "$R/stcmds_breadth_front.txt" 2>/dev/null
echo
echo "############ gate decision + end-to-end (if it ran) ############"
grep -E "gate ->|end-to-end|rows=" "$R/stcmds_breadth.log" | tail -6
[ -s "$R/e2eSTCMDSBREADTH.predictions.jsonl" ] && "$PY" .dsh_checks/restore_eval.py \
  --records "$R/e2eSTCMDSBREADTH.predictions.jsonl" --label "ST-CMDS breadth+fixed e2e" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid"
