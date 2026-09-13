#!/usr/bin/env bash
set -euo pipefail

PART_ROOT=/root/autodl-tmp/datasets/mls_download/parts
SELECT_ROOT=/root/autodl-tmp/datasets/mls_official_cbwhisper_selected
META_ROOT=/root/autodl-tmp/datasets/mls/train/mls_english_opus
LOG_ROOT=/root/autodl-tmp/src/logs/mls_parts
PY=/root/autodl-tmp/great/bin/python
EXPECTED_MD5=60390221eec6f456611563b37f0b052c

mkdir -p "$PART_ROOT" "$SELECT_ROOT" "$LOG_ROOT"

parts=(partaa partab partac partad partae partaf partag)
pids=()
echo "[$(date -Is)] stage=download_parts"
for part in "${parts[@]}"; do
  url="https://dl.fbaipublicfiles.com/mls/mls_english_opus.tar.gz.$part"
  if [[ "$part" == "partaa" || "$part" == "partae" ]]; then
    url="$url?route=alternate-$part"
  fi
  (
    attempt=0
    until aria2c -c -x 8 -s 8 -k 16M --file-allocation=none \
      --summary-interval=30 --console-log-level=notice \
      -d "$PART_ROOT" -o "mls_english_opus.tar.gz.$part" \
      "$url" >> "$LOG_ROOT/$part.log" 2>&1; do
      attempt=$((attempt + 1))
      echo "[$(date -Is)] part=$part retry=$attempt" >> "$LOG_ROOT/$part.log"
      sleep 30
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] error=part_download_failed"
  exit 2
fi

echo "[$(date -Is)] stage=verify_md5"
actual_md5=$(cat "$PART_ROOT"/mls_english_opus.tar.gz.part* | md5sum | awk '{print $1}')
if [[ "$actual_md5" != "$EXPECTED_MD5" ]]; then
  echo "[$(date -Is)] error=md5_mismatch expected=$EXPECTED_MD5 actual=$actual_md5"
  exit 3
fi

member_list="$SELECT_ROOT/upstream_uttid_tar_members.txt"
"$PY" - "$META_ROOT/uttid" "$member_list" <<'PY'
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
members = []
for line in source.read_text(encoding="utf-8-sig").splitlines():
    code = line.strip().split()[0] if line.strip() else ""
    if not code:
        continue
    speaker, book, _ = code.split("_", 2)
    members.append(f"mls_english_opus/train/audio/{speaker}/{book}/{code}.opus")
target.write_text("\n".join(members) + "\n", encoding="utf-8")
print({"members": len(members), "target": str(target)})
PY

echo "[$(date -Is)] stage=extract_selected"
cat "$PART_ROOT"/mls_english_opus.tar.gz.part* \
  | tar -xzf - -C "$SELECT_ROOT" -T "$member_list"

selected="$SELECT_ROOT/mls_english_opus/train"
count=$(find "$selected/audio" -type f -name '*.opus' | wc -l)
expected=$(grep -cve '^$' "$META_ROOT/uttid")
if [[ "$count" -ne "$expected" ]]; then
  echo "[$(date -Is)] error=selected_count_mismatch expected=$expected actual=$count"
  exit 4
fi

cp -a "$META_ROOT"/*.tsv "$META_ROOT"/*.txt "$META_ROOT/uttid" "$selected/"
mkdir -p "$selected/hs" \
  "$selected/keywords-audios/tts" "$selected/keywords-audios/natural" \
  "$selected/keywords-hs/tts" "$selected/keywords-hs/natural"

echo "[$(date -Is)] stage=remove_parts"
resolved=$(readlink -f "$PART_ROOT")
if [[ "$resolved" != "/root/autodl-tmp/datasets/mls_download/parts" ]]; then
  echo "unsafe part root: $resolved"
  exit 5
fi
rm -f -- "$PART_ROOT"/mls_english_opus.tar.gz.part*

cat > "$SELECT_ROOT/extraction_summary.json" <<JSON
{
  "source": "official MLS English Opus archive",
  "archive_md5": "$actual_md5",
  "selected_utterances": $count,
  "metadata_root": "$META_ROOT",
  "audio_root": "$selected/audio"
}
JSON
echo "[$(date -Is)] stage=complete selected=$count"
