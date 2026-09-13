#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
ADAPTER_ROOT="$ROOT/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817"
DATA="$ROOT/data/processed/stcmds_chinesehp_sensevoice"
OUT="$ROOT/outputs/qwen35_9b_stcmds_checkpoint_selection_20260820"

mkdir -p "$OUT/screen" "$OUT/full_dev" "$OUT/test"
cd "$ROOT"

DEV="$DATA/dev.qwen.jsonl"
TEST="$DATA/test.qwen.jsonl"
SCREEN="$OUT/dev_screen_stride4.jsonl"
awk 'NR % 4 == 1' "$DEV" > "$SCREEN"
echo "[$(date '+%F %T')] START dev_rows=$(wc -l < "$DEV") screen_rows=$(wc -l < "$SCREEN") test_rows=$(wc -l < "$TEST")"

checkpoints=(4300 8600 12900 17200 21500 25800 30100 34400 34546)
for step in "${checkpoints[@]}"; do
  pred="$OUT/screen/checkpoint-${step}.predictions.jsonl"
  summary="$OUT/screen/checkpoint-${step}.summary.json"
  if [[ ! -s "$summary" ]]; then
    echo "[$(date '+%F %T')] SCREEN_START checkpoint=$step"
    "$PY" scripts/infer_lora_text.py \
      --input "$SCREEN" --output "$pred" \
      --model-name-or-path "$BASE" --adapter-path "$ADAPTER_ROOT/checkpoint-$step" \
      --batch-size 8 --max-new-tokens 128 --progress-every 64 --disable-thinking
    "$PY" scripts/evaluate_correction_jsonl.py --input "$pred" > "$summary"
    cat "$summary"
    echo "[$(date '+%F %T')] SCREEN_DONE checkpoint=$step"
  fi
done

"$PY" - "$OUT/screen" "$OUT/screen_ranking.tsv" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for path in Path(sys.argv[1]).glob("checkpoint-*.summary.json"):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append((float(metrics["cer"]), int(metrics["worsened_samples"]), path.name.removesuffix(".summary.json")))
rows.sort()
Path(sys.argv[2]).write_text("".join(f"{cer}\t{worse}\t{checkpoint}\n" for cer, worse, checkpoint in rows), encoding="utf-8")
PY
head -3 "$OUT/screen_ranking.tsv" | cut -f3 > "$OUT/top3_checkpoints.txt"
echo "[$(date '+%F %T')] SCREEN_RANKING"
cat "$OUT/screen_ranking.tsv"

while read -r checkpoint; do
  pred="$OUT/full_dev/${checkpoint}.predictions.jsonl"
  summary="$OUT/full_dev/${checkpoint}.summary.json"
  if [[ ! -s "$summary" ]]; then
    echo "[$(date '+%F %T')] FULL_DEV_START checkpoint=$checkpoint"
    "$PY" scripts/infer_lora_text.py \
      --input "$DEV" --output "$pred" \
      --model-name-or-path "$BASE" --adapter-path "$ADAPTER_ROOT/$checkpoint" \
      --batch-size 8 --max-new-tokens 128 --progress-every 128 --disable-thinking
    "$PY" scripts/evaluate_correction_jsonl.py --input "$pred" > "$summary"
    cat "$summary"
    echo "[$(date '+%F %T')] FULL_DEV_DONE checkpoint=$checkpoint"
  fi
done < "$OUT/top3_checkpoints.txt"

"$PY" - "$OUT/full_dev" "$OUT/full_dev_ranking.tsv" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for path in Path(sys.argv[1]).glob("checkpoint-*.summary.json"):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append((float(metrics["cer"]), int(metrics["worsened_samples"]), path.name.removesuffix(".summary.json")))
rows.sort()
Path(sys.argv[2]).write_text("".join(f"{cer}\t{worse}\t{checkpoint}\n" for cer, worse, checkpoint in rows), encoding="utf-8")
PY
best_checkpoint=$(head -1 "$OUT/full_dev_ranking.tsv" | cut -f3)
echo "$best_checkpoint" > "$OUT/best_checkpoint.txt"
echo "[$(date '+%F %T')] BEST_CHECKPOINT=$best_checkpoint"
cat "$OUT/full_dev_ranking.tsv"

test_pred="$OUT/test/${best_checkpoint}.predictions.jsonl"
test_summary="$OUT/test/${best_checkpoint}.summary.json"
echo "[$(date '+%F %T')] TEST_START checkpoint=$best_checkpoint"
"$PY" scripts/infer_lora_text.py \
  --input "$TEST" --output "$test_pred" \
  --model-name-or-path "$BASE" --adapter-path "$ADAPTER_ROOT/$best_checkpoint" \
  --batch-size 8 --max-new-tokens 128 --progress-every 128 --disable-thinking
"$PY" scripts/evaluate_correction_jsonl.py --input "$test_pred" > "$test_summary"
cat "$test_summary"

"$PY" - "$best_checkpoint" "$OUT/full_dev/${best_checkpoint}.summary.json" "$test_summary" "$OUT/final_summary.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path

payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "best_checkpoint": sys.argv[1],
    "dev": json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")),
    "test": json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
}
Path(sys.argv[4]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
echo "[$(date '+%F %T')] ALL_DONE"
