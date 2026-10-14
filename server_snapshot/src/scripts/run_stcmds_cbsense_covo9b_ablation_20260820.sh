#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
SRC=/root/autodl-tmp/src
PY=/root/autodl-tmp/great/bin/python
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
ADAPTER="$ROOT/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
OUT="$ROOT/outputs/qwen35_9b_stcmds_cbsense_ablation_20260820"

mkdir -p "$OUT"
cd "$ROOT"

run_route() {
  local name="$1"
  local input="$2"
  local pred="$OUT/${name}.predictions.jsonl"
  local summary="$OUT/${name}.summary.json"
  echo "[$(date '+%F %T')] START route=$name rows=$(wc -l < "$input")"
  "$PY" scripts/infer_lora_text.py \
    --input "$input" --output "$pred" \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" \
    --batch-size 8 --max-new-tokens 128 --progress-every 128 --disable-thinking
  "$PY" scripts/evaluate_correction_jsonl.py --input "$pred" > "$summary"
  cat "$summary"
  echo "[$(date '+%F %T')] DONE route=$name"
}

run_route error_targeted_chinesehp \
  "$SRC/logs/stcmds_cb_sensevoice_targeted_chinesehp_messages_20260728.jsonl"
run_route standard3139_chinesehp \
  "$SRC/logs/stcmds_cb_sensevoice_standard3139_chinesehp_messages_full_20260729.jsonl"

"$PY" - "$OUT" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "adapter": "checkpoint-34400",
    "naked_9b": {
        "cer": 0.044994035218916366,
        "baseline_cer": 0.05640724320282036,
    },
    "direct_cbsense_error_targeted": {"cer": 0.045446971695251775},
    "historical_4b_error_targeted_chinesehp": {"cer": 0.04542136281893774, "baseline_cer": 0.044833787368908355},
    "historical_4b_standard3139_chinesehp": {"cer": 0.05275715328597119, "baseline_cer": 0.0579206951195627},
    "qwen9b_error_targeted_chinesehp": json.loads((out / "error_targeted_chinesehp.summary.json").read_text(encoding="utf-8")),
    "qwen9b_standard3139_chinesehp": json.loads((out / "standard3139_chinesehp.summary.json").read_text(encoding="utf-8")),
}
(out / "final_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

cat "$OUT/final_summary.json"
echo "[$(date '+%F %T')] ALL_DONE"
