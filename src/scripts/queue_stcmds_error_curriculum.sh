#!/usr/bin/env bash
set -euo pipefail

current_pid=${1:?usage: queue_stcmds_error_curriculum.sh CURRENT_PIPELINE_PID}
workspace=/root/autodl-tmp
completion_metric="$workspace/src/logs/stcmds_chinesehp_covo_hardbalanced_test_metrics_20260720.json"

while kill -0 "$current_pid" 2>/dev/null; do
  sleep 120
done

if [[ ! -s "$completion_metric" ]]; then
  echo "Current pipeline ended without $completion_metric; refusing to start continuation." >&2
  exit 1
fi

exec "$workspace/src/scripts/run_stcmds_covo_error_curriculum.sh"
