#!/usr/bin/env bash
# Wait for the whole queue and print every end-to-end table it produced.
R=/root/autodl-tmp/.dsh_checks/rerank
PY=/root/autodl-tmp/great/bin/python
A=/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev
T=/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test
S=/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test
cd /root/autodl-tmp
for i in $(seq 1 900); do
  [ -f "$R/QUEUE_DONE" ] && break
  sleep 60
done

echo "############ step milestones ############"
grep -E "^\[.*(step |infer DONE|infer FAILED|eval done|SKIP|breadth finished|samecode finished)" "$R/queue_run.log"
echo
arm() {
  local lbl="$1" rec="$2" d="$3"
  if [ -s "$rec" ]; then
    "$PY" .dsh_checks/restore_eval.py --records "$rec" --label "$lbl" \
      --aligned "$d/uttid" --uttid-file "$d/uttid" 2>/dev/null | tail -6
  else
    echo "# $lbl  (missing: $rec)"
  fi
  echo
}
echo "############ end-to-end, FIXED rescorer ############"
arm "AISHELL dev  relaxed + FIXED"          "$R/armG1.predictions.jsonl"                "$A"
arm "THCHS-30    relaxed + FIXED"           "$R/e2eTHCHSFIX.predictions.jsonl"           "$T"
arm "ST-CMDS     orig + FIXED (decisive)"   "$R/e2eSTCMDSORIGFIX.predictions.jsonl"      "$S"
arm "ST-CMDS     relaxed + FIXED"           "$R/e2eSTCMDSFIX.predictions.jsonl"          "$S"
arm "ST-CMDS     breadth-only + FIXED"      "$R/e2eSTCMDSBREADTH.predictions.jsonl"      "$S"
echo "############ reference points ############"
cat <<'REF'
  AISHELL baseline (orig, anchor ON)      2.5922% / 92.65% / destroyed 1
  AISHELL relax, no anchor, no rescorer   2.5495% / 94.82% / destroyed 1
  THCHS   baseline (same code, anchor ON) 3.2241% / 91.27% / destroyed 162
  THCHS   relax + shipped rescorer        3.1674% / 93.96% / destroyed 110
  STCMDS  baseline (orig, anchor ON)      5.0300% / 93.48% / destroyed 0
  STCMDS  orig + shipped rescorer         6.2657% / 92.90% / destroyed 0
  STCMDS  relax + shipped rescorer        5.1279% / 94.00% / destroyed 0
REF
echo
echo "############ breadth experiment (if it ran) ############"
tail -14 "$R/stcmds_breadth.log" 2>/dev/null
echo
echo "############ same-code ST-CMDS baseline (if it ran) ############"
awk '/ST-CMDS held-out: 07-29 evidence/,0' "$R/samecode.log" 2>/dev/null | head -8
