#!/usr/bin/env bash
set -u

TRAIN_PID=176404
WRAPPER_PID=176398
OUTPUT_DIR=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817
WATCH_LOG=/root/autodl-tmp/src/logs/shutdown_after_stcmds_train_20260820.log

echo "[$(date '+%F %T')] watcher started; train_pid=${TRAIN_PID}" >> "$WATCH_LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

sleep 30
if [[ -s "$OUTPUT_DIR/adapter_model.safetensors" && -s "$OUTPUT_DIR/adapter_config.json" ]]; then
  echo "[$(date '+%F %T')] training completed and final adapter exists; shutting down in 60 seconds" >> "$WATCH_LOG"
  sleep 60
  /sbin/shutdown -h now
else
  echo "[$(date '+%F %T')] training exited without final adapter; machine left running for inspection (wrapper_pid=${WRAPPER_PID})" >> "$WATCH_LOG"
  exit 1
fi
