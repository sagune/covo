#!/usr/bin/env bash
set -u

installer_pid=16071
install_log=/root/autodl-tmp/logs/install_librispeech_20260811.log
watch_log=/root/autodl-tmp/logs/shutdown_after_librispeech_20260811.log
dataset_dir=/root/autodl-tmp/datasets/librispeech/LibriSpeech

echo "[$(date '+%F %T')] waiting for installer PID $installer_pid" >> "$watch_log"
while kill -0 "$installer_pid" 2>/dev/null; do
  sleep 60
done

required_splits=(train-clean-100 dev-clean dev-other test-clean test-other)
for split in "${required_splits[@]}"; do
  if [[ ! -d "$dataset_dir/$split" ]]; then
    echo "[$(date '+%F %T')] install incomplete: missing $split; machine remains on" >> "$watch_log"
    exit 1
  fi
done

if ! grep -q 'LibriSpeech installation complete' "$install_log"; then
  echo "[$(date '+%F %T')] completion marker missing; machine remains on" >> "$watch_log"
  exit 1
fi

echo "[$(date '+%F %T')] verified all splits; shutting down" >> "$watch_log"
sync
/sbin/shutdown -h now
