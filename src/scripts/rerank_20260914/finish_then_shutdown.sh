#!/usr/bin/env bash
# Wait for the likelihood-scoring arm, run the MBR analysis on its output, save + push, then
# power the instance off.  Ordered so that NOTHING is lost by the shutdown: the analysis is
# written to disk and committed BEFORE the machine goes down, and the shutdown is the last
# statement in the script.
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
OUT="$R/train_aishell_v1"
PRED="$OUT/stcmds_likelihood.predictions.jsonl"
LOG="$R/finish_shutdown.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "================ FINISH -> SHUTDOWN WATCHER START ================"

# 1. wait for the scoring arm's own wrapper to finish (it moves .inprogress and evaluates)
for i in $(seq 1 360); do            # up to 6 h
  [ -f "$R/LIKELIHOOD_DONE" ] && break
  pgrep -f run_likelihood_arm.sh >/dev/null || { say "likelihood wrapper gone"; break; }
  sleep 60
done
say "wrapper settled: LIKELIHOOD_DONE=$([ -f "$R/LIKELIHOOD_DONE" ] && echo yes || echo no)"
say "scorer output rows=$(wc -l < "$PRED" 2>/dev/null || echo 0)"

# 2. the scorer's accuracy / keep-rate / CER, and the MBR variant over its own scores
if [ -s "$PRED" ]; then
  {
    echo "################ likelihood scorer (zero-shot selector) ################"
    "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$PRED" --label "ST-CMDS likelihood scorer" 2>&1
    echo
    "$PY" "$WS/.dsh_checks/diagnostic_what_can_pick.py" --predictions "$PRED" 2>&1 | head -40
    echo
    echo "################ MBR selection over the scorer's own candidate scores ################"
    "$PY" "$WS/.dsh_checks/mbr_select.py" --records "$PRED" 2>&1
  } > "$R/FINAL_scorer_and_mbr.txt" 2>&1
  say "wrote FINAL_scorer_and_mbr.txt"
  # append the same content into the results doc so it survives in git
  {
    echo
    echo "---"
    echo
    echo "## 13. 打分臂结果与 MBR 选择（关机前的最后一段）"
    echo
    echo '```'
    cat "$R/FINAL_scorer_and_mbr.txt"
    echo '```'
  } >> "$WS/src/RESULTS_RERANK_20260914.md"
else
  say "no scorer output; skipping the analysis"
fi

# 3. commit and push BEFORE powering off
cd "$WS" || exit 1
cp "$R/FINAL_scorer_and_mbr.txt" src/ 2>/dev/null || true
git add -A src/RESULTS_RERANK_20260914.md src/FINAL_scorer_and_mbr.txt 2>/dev/null
git commit -q -m "results: zero-shot likelihood selector and MBR selection; final numbers before shutdown" 2>&1 | tail -2 >> "$LOG"
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git push covo HEAD:cb-sensevoice-covo >> "$LOG" 2>&1
say "git push rc=$?"
git log --oneline -1 >> "$LOG"

# 4. record a final state line, then power off
{
  echo "final state $(date -Is)"
  echo "  git: $(git log --oneline -1)"
  echo "  golden numbers: best ST-CMDS = 4.9356% (adapter V-C); SFT run = 7.3109%; target 4.5000% not reached"
  echo "  next: read src/RUNBOOK_backend_result.md, and RESULTS sections 11-13"
} >> "$LOG"
say "============ POWERING OFF ============"
sync
sleep 5
shutdown -h now 2>/dev/null || poweroff 2>/dev/null || halt 2>/dev/null
