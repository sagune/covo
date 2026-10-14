#!/usr/bin/env bash
# Pre-flight for the AC-COVO evaluation phases (P4 checkpoint selection, P5 transfer).
#
# P4/P5 run unattended hours after they are written, and a single missing or
# TRUNCATED input silently degrades or voids them.  That is not hypothetical here:
# the THCHS eval link was once written as 216 rows instead of 2495 because the
# pipeline waited for "non-empty" instead of "complete", and every row of the
# truncated file still matched perfectly.
#
# So this checks presence AND row counts, and is safe to re-run at any time.
# Exit code 0 iff nothing is missing.
set -uo pipefail
cd /root/autodl-tmp
R=.dsh_checks/rerank
OUT=$R/acoustic_v1
A=datasets/aishell/data_aishell_sensevoice/hotword/dev
T=datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test
S=datasets/stcmds/cb_sensevoice_heldout/hotword/test
B=cbwhisper_covo_migration_20260609_tar_extracted
ok=0; bad=0

# name, path, minimum expected rows (0 = existence only)
chk() {
  local name="$1" path="$2" want="${3:-0}" n
  if [[ ! -s "$path" ]]; then
    printf "  MISSING  %-46s\n" "$name"; bad=$((bad+1)); return
  fi
  n=$(wc -l < "$path" 2>/dev/null || echo 0)
  if [[ "$want" -gt 0 && "$n" -lt "$want" ]]; then
    printf "  TRUNC    %-46s %s < %s\n" "$name" "$n" "$want"; bad=$((bad+1)); return
  fi
  printf "  OK       %-46s %s\n" "$name" "$n"; ok=$((ok+1))
}

echo "--- eval prompts (P4/P5 inputs) ---"
chk "eval_aishell.messages.ac.jsonl" "$OUT/eval_aishell.messages.ac.jsonl" 1334
chk "eval_thchs.messages.ac.jsonl"   "$OUT/eval_thchs.messages.ac.jsonl"   2495
chk "e2eSTCMDS_VA.messages.ac.jsonl" "$R/e2eSTCMDS_VA.messages.ac.jsonl"   5130
echo "--- acoustic stores ---"
chk "ac_aishell_dev.jsonl" "$R/ac_aishell_dev.jsonl" 1334
chk "ac_aishell_dev.bin"   "$R/ac_aishell_dev.bin"
chk "ac_stcmds.jsonl"      "$R/ac_stcmds.jsonl"      5130
chk "ac_stcmds.bin"        "$R/ac_stcmds.bin"
chk "ac_thchs30.jsonl"     "$R/ac_thchs30.jsonl"     2495
chk "ac_thchs30.bin"       "$R/ac_thchs30.bin"
echo "--- aligned / uttid ---"
chk "aishell dev aligned.txt" "$A/aligned.txt"
chk "aishell dev uttid"       "$A/uttid" 1334
chk "thchs aligned.txt"       "$T/aligned.txt"
chk "thchs uttid"             "$T/uttid" 2495
chk "stcmds aligned.txt"      "$S/aligned.txt"
chk "stcmds uttid"            "$S/uttid" 5130
echo "--- code + model ---"
chk "infer_acoustic_covo.py"   "$B/covo/scripts/infer_acoustic_covo.py"
chk "acoustic_conditioning.py" "$B/covo/src/acoustic_conditioning.py"
chk "model config.json"        "$B/models/Qwen3.5-9B/config.json"
chk "restore_eval.py"          ".dsh_checks/restore_eval.py"
chk "edit_precision.py"        ".dsh_checks/edit_precision.py"
chk "ac_gate_report.py"        "src/analysis/ac_gate_report.py"
chk "final projector"          "$OUT/final/acoustic_projector.pt"
echo
echo "PREFLIGHT: $ok present, $bad missing/truncated"
[[ "$bad" -eq 0 ]] || exit 1
