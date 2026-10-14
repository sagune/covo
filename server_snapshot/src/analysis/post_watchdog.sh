#!/usr/bin/env bash
# Chain the post-run analysis automatically once the pipeline finishes.
#
# WHY THIS EXISTS: the pipeline (P0..P6) is autonomous, but the analysis that turns its
# outputs into a VERDICT -- the six pre-registered gates, the within-model ablation, and
# the transfer using the checkpoint dev actually selected -- was a manual step. If no
# agent is present when training finishes, that analysis simply never happens and the
# night produces raw prediction files instead of an answer.
#
# This waits for the pipeline's completion marker and then runs it, so the outcome is
# readable without anyone driving it. It deliberately does NOT edit the running pipeline
# script (bash reads scripts incrementally; editing one mid-run can corrupt execution).
set -uo pipefail
WS=/root/autodl-tmp
R=$WS/.dsh_checks/rerank
LOG=$R/ac_watchdog.log
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "watchdog up; waiting for pipeline completion"
for i in $(seq 1 2880); do        # up to 24 h at 30 s
  if [[ -f "$R/ACOUSTIC_PIPELINE_DONE" ]]; then
    say "pipeline DONE detected"
    break
  fi
  if [[ -f "$R/ACOUSTIC_FAILED" ]]; then
    say "pipeline FAILED; not running post-analysis (read acoustic_pipeline.log)"
    exit 1
  fi
  sleep 30
done

if [[ ! -f "$R/ACOUSTIC_PIPELINE_DONE" ]]; then
  say "gave up waiting after 24 h"
  exit 2
fi

# give the filesystem a moment to settle, then run the full post sequence
sleep 60
say "starting run_all_post.sh"
bash "$WS/.dsh_checks/run_all_post.sh" >> "$LOG" 2>&1
say "run_all_post.sh finished rc=$?"

# make the verdict easy to find
{
  echo "===== AC-COVO VERDICT (auto-generated $(date -Is)) ====="
  echo
  sed -n '/AC-COVO FINAL REPORT/,/^=\{20,\}$/p' "$R/ac_postrun.log" 2>/dev/null | head -80
  echo
  echo "--- within-model ablation ---"
  grep -a -A6 "ACOUSTIC CHANNEL CONTRIBUTION" "$R/ac_postrun.log" 2>/dev/null | head -12
  echo
  echo "--- best-on-dev transfer ---"
  sed -n '/best-on-dev checkpoint/,$p' "$R/ac_postrun.log" 2>/dev/null | head -40
} > "$R/AC_COVO_VERDICT.txt" 2>&1

say "wrote $R/AC_COVO_VERDICT.txt"
say "watchdog done"
