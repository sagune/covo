#!/usr/bin/env bash
# Power the box off once the ENTIRE chain has finished, so the night runs unattended
# and nothing is billed after the result exists.
#
# Chain to wait for:
#   run_acoustic_pipeline.sh : P3 training -> P4 per-checkpoint dev -> P5 transfer
#                              -> writes ACOUSTIC_PIPELINE_DONE
#   post_watchdog.sh         : sees that marker, runs run_all_post.sh (pre-flight,
#                              final report, within-model ablation, transfer with the
#                              checkpoint dev selected) -> writes AC_COVO_VERDICT.txt
#
# The shutdown is gated on real artefacts existing, not merely on processes exiting:
# a machine that powers off with nothing to show is worse than one left running.
#
# Follows the project's existing precedent (/sbin/shutdown -h now, used by
# src/scripts/shutdown_after_stcmds_train_20260820.sh), including the grace period.
set -uo pipefail
WS=/root/autodl-tmp
R=$WS/.dsh_checks/rerank
LOG=$R/shutdown_watcher.log
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "shutdown watcher up; waiting for the pipeline marker"

# ---- 1. wait for the pipeline to settle (up to 24 h)
state="none"
for i in $(seq 1 2880); do
  if [[ -f "$R/ACOUSTIC_PIPELINE_DONE" ]]; then state="done"; break; fi
  if [[ -f "$R/ACOUSTIC_FAILED" ]];        then state="failed"; break; fi
  sleep 30
done
say "pipeline state: $state"

# ---- 2. if it finished, give the post-analysis time to produce the verdict
if [[ "$state" == "done" ]]; then
  say "waiting for the post-analysis verdict"
  for i in $(seq 1 480); do          # up to 4 h
    [[ -f "$R/AC_COVO_VERDICT.txt" ]] && break
    # if the post-run log shows the watchdog died, stop waiting
    if [[ -f "$R/ac_watchdog.log" ]] && grep -q "watchdog done" "$R/ac_watchdog.log"; then
      say "post watchdog reported done"
      break
    fi
    sleep 30
  done
fi

# ---- 3. summarise what exists before powering off
{
  echo "=== AC-COVO final state $(date -Is) ==="
  echo "pipeline state: $state"
  echo
  echo "checkpoints : $(ls -d "$R"/acoustic_v1/final/checkpoint-* 2>/dev/null | wc -l)"
  echo "final projector present : $([[ -s "$R/acoustic_v1/final/acoustic_projector.pt" ]] && echo yes || echo NO)"
  echo "verdict file present    : $([[ -s "$R/AC_COVO_VERDICT.txt" ]] && echo yes || echo NO)"
  echo "verdict path            : $R/AC_COVO_VERDICT.txt"
  echo "dev metrics rebuilt     : $(ls -la "$R/dev_metrics_rebuilt.tsv" 2>/dev/null | awk '{print $5" bytes"}' || echo none)"
  echo
  echo "--- last lines of the pipeline log ---"
  tail -15 "$R/acoustic_pipeline.log" 2>/dev/null
} > "$R/AC_COVO_FINAL_STATE.txt" 2>&1
say "wrote AC_COVO_FINAL_STATE.txt"

have_result=0
[[ -s "$R/AC_COVO_VERDICT.txt" ]] && have_result=1
[[ "$state" == "done" && -s "$R/acoustic_v1/final/acoustic_projector.pt" ]] && have_result=1
[[ "$state" == "failed" ]] && have_result=1     # a failure IS the night's result

if [[ "$have_result" -eq 1 ]]; then
  say "result artefacts in place; powering off in 120 s (^C would need a login, which is fine)"
  sync
  sleep 120
  sync
  /sbin/shutdown -h now
else
  say "NO usable result artefacts; leaving the machine running for inspection"
  exit 1
fi
