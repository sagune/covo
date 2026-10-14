#!/usr/bin/env bash
# Wait for the decisive arm (step 1) and print its three-way comparison.
R=/root/autodl-tmp/.dsh_checks/rerank
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
PY=/root/autodl-tmp/great/bin/python
cd /root/autodl-tmp
for i in $(seq 1 300); do
  [ -f "$R/STCMDS_ORIG_FIXED_E2E_DONE" ] && break
  sleep 60
done

echo "############ ST-CMDS original admission: FIXED rescorer ############"
awk '/ST-CMDS orig\+fixed e2e/,0' "$R/queue_run.log" | head -8
echo
echo "############ the three-way comparison ############"
if [ -s "$R/e2eSTCMDSORIGFIX.predictions.jsonl" ]; then
  "$PY" .dsh_checks/restore_eval.py --records "$R/e2eSTCMDSORIGFIX.predictions.jsonl" \
    --label "orig admission + FIXED rescorer" --aligned "$S/aligned.txt" --uttid-file "$S/uttid"
fi
echo "  reference points (already measured):"
echo "    baseline, orig admission + anchor ON   5.0300% / 93.48% / destroyed 0"
echo "    orig admission + SHIPPED rescorer      6.2657% / 92.90% / destroyed 0"
echo
echo "############ queue milestones so far ############"
grep -E "^\[.*(step|infer DONE|eval done|SKIP|FAILED)" "$R/queue_run.log" | tail -12
