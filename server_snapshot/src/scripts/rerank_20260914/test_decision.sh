#!/usr/bin/env bash
# Test the sequencer's DECIDE branch against synthetic logs.
#
# This branch is the most consequential untested logic in the pipeline: it decides whether
# to spend ~1h on the likelihood arm and ~3.5h on DPO.  A wrong answer either burns that
# time on a run that already met the target, or skips the fallback on one that missed.
#
# It uses the REAL get_cer.py and the REAL branch expressions copied from
# run_night_sequencer.sh, against three synthetic logs:
#   A. trained adapter at 4.4000  -> TARGET MET   (NEED_DPO=0)
#   B. trained adapter at 4.7000  -> TARGET MISSED (NEED_DPO=1)
#   C. no trained-adapter block   -> treated as MISSED
set -uo pipefail
WS=/root/autodl-tmp
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
TMP=/tmp/dectest
TARGET=4.50
BASELINE=4.9356
rm -rf "$TMP"; mkdir -p "$TMP"

mklog() {  # file, cer-or-empty
  local f="$1" cer="$2"
  : > "$f"
  echo "# AISHELL dev, final" >> "$f"
  echo "rows 1334 | reference chars 21102" >> "$f"
  echo "  restore[deployable]    CER  2.4123% | recall 95.20% (1270/1334) destroyed   0 | edited  300 | beyond-N-best +0.180 pp (imp 90 wor 20, n=227)" >> "$f"
  if [ -n "$cer" ]; then
    echo "# ST-CMDS, trained adapter" >> "$f"
    echo "rows 5130 | reference chars 56163" >> "$f"
    echo "  restore[deployable]    CER  ${cer}% | recall 95.90% (1648/1718) destroyed   0 | edited  350 | beyond-N-best +0.190 pp (imp 95 wor 22, n=711)" >> "$f"
  fi
}

read_cer() {  # the sequencer's own helper, verbatim
  "$PY" "$WS/.dsh_checks/get_cer.py" --log "$1" --label-substr "ST-CMDS, trained adapter" \
    | tr ' ' '\n' | grep '^CER=' | cut -d= -f2
}

decide() {  # log -> prints the branch taken
  local TRLOG="$1" CER NEED_DPO
  CER=$(read_cer "$TRLOG")
  NEED_DPO=1
  if [ -z "${CER:-}" ] || [ "${CER:-}" = "none" ]; then
    echo "  branch=MISSED(absent)  CER='${CER:-}'  NEED_DPO=$NEED_DPO"
  elif awk -v c="$CER" -v t="$TARGET" 'BEGIN{exit !(c<=t)}'; then
    NEED_DPO=0
    echo "  branch=MET             CER=$CER  NEED_DPO=$NEED_DPO"
  else
    echo "  branch=MISSED          CER=$CER  NEED_DPO=$NEED_DPO"
  fi
}

FAIL=0
for case in "4.4000|A: 4.4000 is at/below target" \
            "4.5000|B: exactly 4.5000 is AT target (<=)" \
            "4.5001|C: 4.5001 is just above target" \
            "4.7000|D: 4.7000 is above target" \
            "4.9356|E: equal to the baseline" \
            "5.5000|F: worse than the baseline"; do
  cer="${case%%|*}"; label="${case##*|}"
  mklog "$TMP/log_$cer.txt" "$cer"
  echo "$label"
  decide "$TMP/log_$cer.txt"
done
echo "G: no trained-adapter block at all (evaluation missing)"
mklog "$TMP/log_none.txt" ""
decide "$TMP/log_none.txt"

echo
echo "-- independent check of get_cer on an absent label --"
"$PY" "$WS/.dsh_checks/get_cer.py" --log "$TMP/log_none.txt" --label-substr "ST-CMDS, trained adapter"; echo "  rc=$? (1 expected)"
rm -rf "$TMP"
