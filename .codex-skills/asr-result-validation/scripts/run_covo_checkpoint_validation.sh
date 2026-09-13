#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --covo-root DIR --python FILE --model DIR --adapter-root DIR \\" >&2
  echo "  --train-file FILE --dev-file FILE --test-file FILE --output-dir DIR \\" >&2
  echo "  --steps STEP,STEP,... [--screen-stride 4] [--top-k 3] [--batch-size 8] \\" >&2
  echo "  [--max-new-tokens 96]" >&2
}

covo_root=
python_bin=
model=
adapter_root=
train_file=
dev_file=
test_file=
output_dir=
steps_csv=
screen_stride=4
top_k=3
batch_size=8
max_new_tokens=96

while [[ $# -gt 0 ]]; do
  case "$1" in
    --covo-root) covo_root="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --train-file) train_file="$2"; shift 2 ;;
    --dev-file) dev_file="$2"; shift 2 ;;
    --test-file) test_file="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --steps) steps_csv="$2"; shift 2 ;;
    --screen-stride) screen_stride="$2"; shift 2 ;;
    --top-k) top_k="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

required=(covo_root python_bin model adapter_root train_file dev_file test_file output_dir steps_csv)
for name in "${required[@]}"; do
  if [[ -z "${!name}" ]]; then
    echo "Missing required argument: ${name}" >&2
    usage
    exit 2
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
integrity_tool="$script_dir/check_split_integrity.py"
rank_tool="$script_dir/rank_checkpoint_summaries.py"
infer="$covo_root/scripts/infer_lora_text.py"
evaluate="$covo_root/scripts/evaluate_correction_jsonl.py"

for path in "$python_bin" "$train_file" "$dev_file" "$test_file" "$integrity_tool" "$rank_tool" "$infer" "$evaluate"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
[[ -d "$model" ]] || { echo "Missing model directory: $model" >&2; exit 2; }
[[ -d "$adapter_root" ]] || { echo "Missing adapter directory: $adapter_root" >&2; exit 2; }

IFS=',' read -r -a steps <<< "$steps_csv"
for step in "${steps[@]}"; do
  [[ "$step" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint step: $step" >&2; exit 2; }
  [[ -d "$adapter_root/checkpoint-$step" ]] || {
    echo "Missing checkpoint: $adapter_root/checkpoint-$step" >&2
    exit 2
  }
done

mkdir -p "$output_dir/screen" "$output_dir/full_dev" "$output_dir/test"

echo "[$(date -Is)] START"
echo "[$(date -Is)] checkpoints=${steps[*]} screen_stride=$screen_stride top_k=$top_k batch_size=$batch_size max_new_tokens=$max_new_tokens"

"$python_bin" "$integrity_tool" \
  --train "$train_file" --dev "$dev_file" --test "$test_file" \
  --output "$output_dir/split_integrity.json"

screen_file="$output_dir/dev_screen_stride${screen_stride}.jsonl"
if [[ ! -s "$screen_file" ]]; then
  awk -v stride="$screen_stride" 'NR % stride == 1' "$dev_file" > "$screen_file"
fi

"$python_bin" - "$output_dir/run_manifest.json" "$train_file" "$dev_file" "$test_file" \
  "$model" "$adapter_root" "$steps_csv" "$screen_stride" "$top_k" "$batch_size" "$max_new_tokens" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

output, train, dev, test, model, adapter, steps, stride, top_k, batch, max_tokens = sys.argv[1:]
files = {}
for name, raw_path in (("train", train), ("dev", dev), ("test", test)):
    path = Path(raw_path)
    stat = path.stat()
    files[name] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
payload = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "model": str(Path(model).resolve()),
    "adapter_root": str(Path(adapter).resolve()),
    "checkpoints": [int(value) for value in steps.split(",")],
    "selection_rule": ["lowest_corpus_cer", "fewest_worsened_samples", "earlier_step"],
    "screen_stride": int(stride),
    "top_k": int(top_k),
    "batch_size": int(batch),
    "max_new_tokens": int(max_tokens),
    "disable_thinking": True,
    "sort_by_prompt_length": False,
    "inference_resume_supported": False,
    "atomic_prediction_publish": True,
    "files": files,
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

run_inference() {
  local input="$1"
  local checkpoint="$2"
  local prediction="$3"
  local summary="$4"
  local label="$5"

  if [[ -s "$summary" ]]; then
    echo "[$(date -Is)] SKIP completed=$label checkpoint=$checkpoint"
    return
  fi
  if [[ ! -s "$prediction" ]]; then
    local inprogress="${prediction}.inprogress"
    echo "[$(date -Is)] INFER_START stage=$label checkpoint=$checkpoint rows=$(wc -l < "$input")"
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$covo_root/src" \
    "$python_bin" "$infer" \
      --input "$input" --output "$inprogress" \
      --model-name-or-path "$model" --adapter-path "$adapter_root/checkpoint-$checkpoint" \
      --batch-size "$batch_size" --max-new-tokens "$max_new_tokens" \
      --progress-every 256 --disable-thinking
    mv -f "$inprogress" "$prediction"
  else
    echo "[$(date -Is)] REUSE_PREDICTION stage=$label checkpoint=$checkpoint"
  fi
  PYTHONPATH="$covo_root/src" "$python_bin" "$evaluate" --input "$prediction" > "$summary"
  [[ -s "$summary" ]] || { echo "Empty summary: $summary" >&2; exit 1; }
  cat "$summary"
  echo "[$(date -Is)] INFER_DONE stage=$label checkpoint=$checkpoint"
}

for step in "${steps[@]}"; do
  run_inference "$screen_file" "$step" \
    "$output_dir/screen/checkpoint-${step}.predictions.jsonl" \
    "$output_dir/screen/checkpoint-${step}.summary.json" screen
done

"$python_bin" "$rank_tool" "$output_dir/screen" --top-k "$top_k" \
  --ranking "$output_dir/screen_ranking.tsv" \
  --selection "$output_dir/screen_selection.json"

mapfile -t top_checkpoints < <("$python_bin" -c \
  'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["top_checkpoints"]))' \
  "$output_dir/screen_selection.json")

for checkpoint in "${top_checkpoints[@]}"; do
  step="${checkpoint#checkpoint-}"
  run_inference "$dev_file" "$step" \
    "$output_dir/full_dev/checkpoint-${step}.predictions.jsonl" \
    "$output_dir/full_dev/checkpoint-${step}.summary.json" full_dev
done

"$python_bin" "$rank_tool" "$output_dir/full_dev" --top-k 1 \
  --ranking "$output_dir/full_dev_ranking.tsv" \
  --selection "$output_dir/full_dev_selection.json"
best_checkpoint="$("$python_bin" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
  "$output_dir/full_dev_selection.json")"
printf '%s\n' "$best_checkpoint" > "$output_dir/best_checkpoint.txt"
echo "[$(date -Is)] CHECKPOINT_LOCKED=$best_checkpoint"

best_step="${best_checkpoint#checkpoint-}"
run_inference "$test_file" "$best_step" \
  "$output_dir/test/${best_checkpoint}.predictions.jsonl" \
  "$output_dir/test/${best_checkpoint}.summary.json" held_out_test

"$python_bin" - "$output_dir" "$best_checkpoint" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
checkpoint = sys.argv[2]
dev = json.loads((out / "full_dev" / f"{checkpoint}.summary.json").read_text(encoding="utf-8"))
test = json.loads((out / "test" / f"{checkpoint}.summary.json").read_text(encoding="utf-8"))
baseline = float(test["baseline_cer"])
result = float(test["cer"])
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "classification": "full_held_out",
    "best_checkpoint": checkpoint,
    "selection_rule": ["lowest_corpus_cer", "fewest_worsened_samples", "earlier_step"],
    "dev": dev,
    "test": test,
    "test_absolute_cer_change": result - baseline,
    "test_relative_cer_change": ((result - baseline) / baseline) if baseline else None,
    "artifacts": {
        "manifest": str((out / "run_manifest.json").resolve()),
        "split_integrity": str((out / "split_integrity.json").resolve()),
        "screen_ranking": str((out / "screen_ranking.tsv").resolve()),
        "full_dev_ranking": str((out / "full_dev_ranking.tsv").resolve()),
        "prediction": str((out / "test" / f"{checkpoint}.predictions.jsonl").resolve()),
    },
}
(out / "final_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

touch "$output_dir/ALL_DONE"
cat "$output_dir/final_summary.json"
echo "[$(date -Is)] ALL_DONE"
