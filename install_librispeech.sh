#!/usr/bin/env bash
set -euo pipefail

dataset_root=/root/autodl-tmp/datasets/librispeech
mirror_url=https://openslr.trmal.net/resources/12
mkdir -p "$dataset_root"
cd "$dataset_root"

source /etc/network_turbo >/dev/null
wget -q -O md5sum.txt https://www.openslr.org/resources/12/md5sum.txt

for archive in \
  train-clean-100.tar.gz \
  dev-clean.tar.gz \
  dev-other.tar.gz \
  test-clean.tar.gz \
  test-other.tar.gz
do
  echo "[$(date '+%F %T')] downloading $archive"
  aria2c \
    --continue=true \
    --check-certificate=false \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --file-allocation=none \
    --summary-interval=60 \
    "$mirror_url/$archive"
  grep " $archive$" md5sum.txt | md5sum -c -
  echo "[$(date '+%F %T')] extracting $archive"
  tar -xzf "$archive"
  rm -f -- "$archive"
done

echo "[$(date '+%F %T')] LibriSpeech installation complete"
du -sh "$dataset_root/LibriSpeech"
df -h /root/autodl-tmp
