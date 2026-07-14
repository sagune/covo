#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-187202}"
LOG="${LOG:-/root/autodl-tmp/src/logs/shutdown_after_sensevoice_kws_train_20260715.log}"

{
  echo "[watcher-start] $(date -Is) waiting for train pid ${TRAIN_PID}"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[watcher-done] $(date -Is) train pid ${TRAIN_PID} finished"
  sync
  echo "[shutdown] $(date -Is) powering off"
} >> "${LOG}" 2>&1

/sbin/shutdown -h now "SenseVoice KWS training finished"
