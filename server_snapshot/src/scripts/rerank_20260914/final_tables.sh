#!/usr/bin/env bash
# Assemble the full acceptance evidence: front-end and end-to-end, every arm,
# every dataset.  Safe to run repeatedly; arms that have not been produced yet are
# reported as missing instead of failing.
set -uo pipefail

WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
S="$WS/.dsh_checks/stcmds"
PY="$WS/great/bin/python"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
C="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"

arm() {  # label, records, uttid-dir
  local label="$1" rec="$2" d="$3"
  if [[ -s "$rec" ]]; then
    "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$rec" --label "$label" \
      --aligned "$d/aligned.txt" --uttid-file "$d/uttid" 2>&1 | sed -n '1p;5,9p'
  else
    printf '# %s\n  (missing: %s)\n' "$label" "$rec"
  fi
  echo
}

echo "################ AISHELL dev (1334 rows / 599 mentions) — tuning set ################"
arm "baseline (orig admission, anchor ON, shipped rerank)" "$S/../aishell_selector/aishell_9b_aishell_adapter.predictions.jsonl" "$A"
arm "relax + old rerank (armF2)"          "$R/armF2.predictions.jsonl" "$A"
arm "relax + FIXED rerank (armG1)"        "$R/armG1.predictions.jsonl" "$A"

echo "################ THCHS-30 (2495 rows / 14137 mentions) ################"
arm "baseline same-code (shipped cfg)"    "$R/thchs_base_predictions.jsonl" "$T"
arm "relax + old rerank"                  "$R/e2eTHCHS.predictions.jsonl" "$T"
arm "relax + FIXED rerank"                "$R/e2eTHCHSFIX.predictions.jsonl" "$T"

echo "################ ST-CMDS held-out (5130 rows / 1718 mentions) ################"
arm "baseline 7/29 (orig admission, anchor ON, shipped rerank)" "$S/stcmds_9b_stcmds_adapter.predictions.jsonl" "$C"
arm "orig admission + old rerank (safe subset)" "$R/e2eSTCMDS_SAFE.predictions.jsonl" "$C"
arm "orig admission + FIXED rerank"       "$R/e2eSTCMDSORIGFIX.predictions.jsonl" "$C"
arm "relax + old rerank"                  "$R/e2eSTCMDS.predictions.jsonl" "$C"
arm "relax + FIXED rerank"                "$R/e2eSTCMDSFIX.predictions.jsonl" "$C"
arm "breadth-only relax + FIXED rerank"   "$R/e2eSTCMDSBREADTH.predictions.jsonl" "$C"

echo "################ front-end only (fixed reranker, from apply_rerank_v2) ################"
cat "$R/v2_confirm.log" 2>/dev/null | grep -E '^# |front-end CER|designated recall|top-1 changed'
if [[ -s "$R/fixed_rerank_e2e.log" ]]; then
  echo
  echo "---- relaxed-admission fixed rerank (queue) ----"
  grep -E '^# |front-end CER|designated recall|top-1 changed' "$R/fixed_rerank_e2e.log"
fi
if [[ -s "$R/stcmds_breadth_front.txt" ]]; then
  echo
  echo "---- breadth-only ST-CMDS + fixed rerank ----"
  cat "$R/stcmds_breadth_front.txt"
fi
