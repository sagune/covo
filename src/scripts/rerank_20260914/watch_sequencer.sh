#!/usr/bin/env bash
# Watchdog: keep exactly one run_night_sequencer.sh alive until it finishes.
#
# The sequencer is the only thing that produces MORNING_REPORT.txt, and it has to survive
# ~5.5h of training plus up to ~5h of arms unattended.  Nothing restarts it if it dies, so
# a single crash (OOM kill, a bad edit, a stray pkill) would leave the whole night with no
# result.  This loop is the insurance.
#
# How it detects death: the sequencer holds an flock on sequencer.lock for its whole life.
# If the lock is FREE, no sequencer is running.  Poll every 5 minutes - long enough that
# the stale-lock window (a killed sequencer's orphaned `sleep 60` inheriting the fd) has
# always cleared, so the watchdog cannot be fooled into double-starting by that.
#
# Safety: the sequencer itself refuses to start while another holds the lock (`flock -w 120`),
# so even a mistimed restart cannot produce two GPU owners.
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
LOCK="$R/sequencer.lock"
LOG="$R/watchdog.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "watchdog start (pid $$)"
for i in $(seq 1 288); do          # 288 * 5 min = 24 h ceiling
  if [ -f "$R/SEQUENCER_DONE" ]; then
    say "SEQUENCER_DONE present; watchdog exiting"; exit 0
  fi
  # free lock => no sequencer alive
  if flock -n "$LOCK" -c true 2>/dev/null; then
    say "lock is free and SEQUENCER_DONE is absent -> restarting the sequencer"
    cd "$WS" || exit 1
    setsid nohup bash "$WS/.dsh_checks/run_night_sequencer.sh" \
      < /dev/null >> "$R/sequencer.out" 2>&1 &
    disown
    sleep 30
  fi
  sleep 300
done
say "watchdog hit its 24h ceiling; exiting"
