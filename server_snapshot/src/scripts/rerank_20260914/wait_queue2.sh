#!/usr/bin/env bash
# Wait for the v2 queue and print every end-to-end table, learned ranker first.
R=/root/autodl-tmp/.dsh_checks/rerank
PY=/root/autodl-tmp/great/bin/python
A=/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev
T=/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
cd /root/autodl-tmp
for i in $(seq 1 900); do
  [ -f "$R/QUEUE2_DONE" ] && break
  sleep 60
done
echo "############ v2 milestones ############"
grep -E "^\[.*(step |infer DONE|infer FAILED|eval done|SKIP|breadth finished|samecode done)" "$R/queue2_run.log"
echo
arm() {
  if [ -s "$2" ]; then
    "$PY" .dsh_checks/restore_eval.py --records "$2" --label "$1" \
      --aligned "$3/aligned.txt" --uttid-file "$3/uttid" 2>/dev/null | sed -n '1p;5,9p'
  else
    echo "# $1  (missing)"
  fi
  echo
}
echo "############ LEARNED ranker, end to end ############"
arm "ST-CMDS + learned (train on AISHELL)" "$R/e2eSTCMDSLEARN.predictions.jsonl" "$S"
arm "THCHS-30 + learned (train on AISHELL)" "$R/e2eTHCHSLEARN.predictions.jsonl" "$T"
echo "############ FIXED rescorer, end to end ############"
arm "AISHELL dev relaxed + FIXED"  "$R/armG1.predictions.jsonl" "$A"
arm "THCHS-30 relaxed + FIXED"     "$R/e2eTHCHSFIX.predictions.jsonl" "$T"
arm "ST-CMDS orig + FIXED"         "$R/e2eSTCMDSORIGFIX.predictions.jsonl" "$S"
arm "ST-CMDS relaxed + FIXED"      "$R/e2eSTCMDSFIX.predictions.jsonl" "$S"
arm "ST-CMDS breadth + FIXED"      "$R/e2eSTCMDSBREADTH.predictions.jsonl" "$S"
echo "############ reference points ############"
cat <<'REF'
  AISHELL baseline (orig, anchor ON)       2.5922% / 92.65% / destroyed 1
  AISHELL relax no-anchor no-rescorer      2.5495% / 94.82% / destroyed 1
  THCHS   baseline (same code, anchor ON)  3.2241% / 91.27% / destroyed 162
  THCHS   relax + shipped rescorer         3.1674% / 93.96% / destroyed 110
  STCMDS  baseline (orig, anchor ON)       5.0300% / 93.48% / destroyed 0
  STCMDS  orig + shipped rescorer          6.2657% / 92.90% / destroyed 0
  STCMDS  relax + shipped rescorer         5.1279% / 94.00% / destroyed 0
REF
echo
echo "############ breadth experiment ############"
tail -14 "$R/stcmds_breadth.log" 2>/dev/null
echo
echo "############ same-code ST-CMDS baseline ############"
awk '/ST-CMDS held-out: 07-29 evidence/,0' "$R/samecode.log" 2>/dev/null | head -8
