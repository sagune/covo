#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/root/autodl-tmp}"
python="${PYTHON_BIN:-${workspace}/great/bin/python}"
covo_root="${workspace}/covo"
adapter_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_magicdata_hardneg_dropout_balanced_2epoch_20260805"
model="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B"
test_input="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/magicdata_original_covo/test.hardneg.qwen.jsonl"
log_root="${workspace}/src/logs"
steps=(2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 22392)

echo "[$(date -Is)] waiting for full dev checkpoint evaluation"
while true; do
  complete=0
  for step in "${steps[@]}"; do
    [[ -s "${log_root}/magicdata_original_covo_dev_ckpt${step}_summary_20260807.json" ]] && complete=$((complete + 1))
  done
  echo "[$(date -Is)] dev summaries=${complete}/${#steps[@]}"
  [[ "${complete}" -eq "${#steps[@]}" ]] && break
  if ! pgrep -f "infer_lora_text.py.*magicdata_original_covo/dev.hardneg.qwen.jsonl" >/dev/null; then
    echo "[error] dev evaluation stopped with ${complete}/${#steps[@]} summaries" >&2
    exit 1
  fi
  sleep 120
done

selection="${log_root}/magicdata_original_covo_dev_selection_20260807.json"
"${python}" - "${selection}" "${log_root}" "${steps[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
log_root = Path(sys.argv[2])
rows = []
for step in map(int, sys.argv[3:]):
    path = log_root / f"magicdata_original_covo_dev_ckpt{step}_summary_20260807.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"step": step, **metrics})
best = min(rows, key=lambda row: (float(row["cer"]), row["step"]))
output.write_text(json.dumps({"best": best, "checkpoints": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(best["step"])
PY
best_step="$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["step"])' "${selection}")"

echo "[$(date -Is)] selected checkpoint-${best_step}; starting one-shot test"
prediction="${log_root}/magicdata_original_covo_test_ckpt${best_step}_predictions_20260807.jsonl"
PYTHONPATH="${covo_root}/src" CUDA_VISIBLE_DEVICES=0 "${python}" "${covo_root}/scripts/infer_lora_text.py" \
  --input "${test_input}" --output "${prediction}" \
  --model-name-or-path "${model}" --adapter-path "${adapter_root}/checkpoint-${best_step}" \
  --batch-size 32 --max-new-tokens 96 --progress-every 800 \
  --disable-thinking --sort-by-prompt-length --resume
PYTHONPATH="${covo_root}/src" "${python}" "${covo_root}/scripts/evaluate_correction_jsonl.py" \
  --input "${prediction}" > "${log_root}/magicdata_original_covo_test_ckpt${best_step}_summary_20260807.json"
echo "[$(date -Is)] test complete for checkpoint-${best_step}"
