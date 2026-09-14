#!/usr/bin/env bash
# CPU-only dry run of the downstream half of the pipeline: render the bridge
# messages for the fixed-rerank products so that (a) the format is validated
# before any GPU is spent, and (b) the paused end-to-end queue resumes straight at
# the model step.  No model is loaded here.
set -uo pipefail

WS=/root/autodl-tmp
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
LOG="$R/bridge_dryrun.log"
: > "$LOG"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

cd "$WS"
bridge() {  # evidence, out-messages, expected rows
  local ev="$1" msg="$2" want="$3"
  if [[ ! -s "$ev" ]]; then
    echo "SKIP  $(basename "$ev") -- missing (needs GPU to build)" | tee -a "$LOG"
    return
  fi
  if [[ -s "$msg" ]]; then
    echo "HAVE  $(basename "$msg")" | tee -a "$LOG"
  else
    "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$ev" \
      --output "$msg.inprogress" "${COMMON[@]}" >> "$LOG" 2>&1
    if [[ -s "$msg.inprogress" ]]; then mv -f "$msg.inprogress" "$msg"; fi
  fi
  local n
  n=$(wc -l < "$msg" 2>/dev/null || echo 0)
  if [[ "$n" == "$want" ]]; then
    echo "OK    $(basename "$msg")  rows=$n" | tee -a "$LOG"
  else
    echo "BAD   $(basename "$msg")  rows=$n (expected $want)" | tee -a "$LOG"
  fi
}

bridge "$R/dev_relax_fixed.jsonl"     "$R/armG1.messages.jsonl"            1334
bridge "$R/thchs_relax_fixed.jsonl"   "$R/e2eTHCHSFIX.messages.jsonl"      2495
bridge "$R/stcmds_orig_fixed.jsonl"   "$R/e2eSTCMDSORIGFIX.messages.jsonl" 5130
bridge "$R/stcmds_relax_fixed.jsonl"  "$R/e2eSTCMDSFIX.messages.jsonl"     5130

echo | tee -a "$LOG"
echo "=== rendered-prompt sanity check (AISHELL armG1, first row) ===" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
p = Path("/root/autodl-tmp/.dsh_checks/rerank/armG1.messages.jsonl")
if not p.exists() or p.stat().st_size == 0:
    print("(armG1 messages not built)"); raise SystemExit
r = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
msgs = r.get("messages") or []
content = msgs[1].get("content") if len(msgs) > 1 else (msgs[0].get("content") if msgs else "")
if isinstance(content, list):
    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
print("keys:", sorted(r.keys()))
print("prompt chars:", len(content))
print("---- first 1400 chars ----")
print(content[:1400])
PY
touch "$R/BRIDGE_DRYRUN_DONE"
