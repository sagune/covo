#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/root/autodl-tmp
SRC="$WORKSPACE/src"
PY="$WORKSPACE/great/bin/python"
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_magicdata_hardneg_balanced_1epoch_20260820/checkpoint-39185"
SENSEVOICE=/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master
OUT="$COVO/outputs/qwen35_9b_magicdata_cbsense_oraclelex6000_test_20260825"

EVIDENCE="$OUT/cb_sensevoice_test.evidence.jsonl"
MESSAGES="$OUT/cb_sensevoice_test.messages.jsonl"
PREDICTIONS="$OUT/checkpoint-39185.predictions.jsonl"
SUMMARY="$OUT/checkpoint-39185.summary.json"

mkdir -p "$OUT"

echo "[$(date -Is)] START classification=leaked_diagnostic"
echo "[$(date -Is)] protocol=CB-SenseVoice_oraclelex6000_test_plus_locked_Qwen9B_checkpoint-39185"

if [[ ! -s "$EVIDENCE" ]]; then
  echo "[$(date -Is)] CBSENSE_START rows=24279"
  cd "$SRC"
  CBW_EVIDENCE_ONLY=1 \
  CBW_EVIDENCE_OUT="$EVIDENCE" \
  CBW_EVAL_BOOTSTRAPS=0 \
  CBW_METRICS_OUT="$OUT/cb_sensevoice_test.lightning_metrics.csv" \
  CBW_DEBUG_LOG="$OUT/cb_sensevoice_test.runtime.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-magicdata.yaml \
    --data.init_args.test_split=test \
    --model.init_args.split=test \
    --model.init_args.sensevoice_ckpt="$SENSEVOICE" \
    --model.init_args.oracle_nbest_detail_path="$OUT/cb_sensevoice_test.oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$OUT/cb_sensevoice_test.oracle_summary.csv"
  echo "[$(date -Is)] CBSENSE_DONE rows=$(wc -l < "$EVIDENCE")"
else
  echo "[$(date -Is)] CBSENSE_REUSE rows=$(wc -l < "$EVIDENCE")"
fi

[[ "$(wc -l < "$EVIDENCE")" -eq 24279 ]] || {
  echo "[error] CB-SenseVoice evidence row count is not 24279" >&2
  exit 1
}

if [[ ! -s "$MESSAGES" ]]; then
  echo "[$(date -Is)] BRIDGE_START"
  cd "$WORKSPACE"
  "$PY" "$SRC/analysis/cbsensevoice_covo_bridge.py" prepare \
    --input "$EVIDENCE" --output "$MESSAGES" \
    --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 \
    --hotword-source prompt --include-pinyin --prompt-mode selector \
    --protect-supported-hotwords --clean-nbest --include-consensus-spans \
    --include-hotword-evidence --prefer-same-length --trust-asr-top1 \
    --preserve-anchor-digits
  echo "[$(date -Is)] BRIDGE_DONE rows=$(wc -l < "$MESSAGES")"
else
  echo "[$(date -Is)] BRIDGE_REUSE rows=$(wc -l < "$MESSAGES")"
fi

[[ "$(wc -l < "$MESSAGES")" -eq 24279 ]] || {
  echo "[error] bridge message row count is not 24279" >&2
  exit 1
}

if [[ ! -s "$SUMMARY" ]]; then
  if [[ ! -s "$PREDICTIONS" ]]; then
    echo "[$(date -Is)] QWEN9B_START rows=24279 checkpoint=39185"
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" \
    "$PY" "$COVO/scripts/infer_lora_text.py" \
      --input "$MESSAGES" --output "$PREDICTIONS.inprogress" \
      --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking
    mv -f "$PREDICTIONS.inprogress" "$PREDICTIONS"
    echo "[$(date -Is)] QWEN9B_DONE rows=$(wc -l < "$PREDICTIONS")"
  fi
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/evaluate_correction_jsonl.py" \
    --input "$PREDICTIONS" > "$SUMMARY"
fi

"$PY" - "$OUT" "$SUMMARY" "$ADAPTER" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
metrics = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
baseline = float(metrics["baseline_cer"])
result = float(metrics["cer"])
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "classification": "leaked_diagnostic",
    "leakage_reason": "CB-SenseVoice 6000-word split-specific context vocabulary was extracted from MAGICDATA test references",
    "system": "CB-SenseVoice oraclelex6000 + Qwen3.5-9B MAGICDATA LoRA",
    "locked_adapter": str(Path(sys.argv[3]).resolve()),
    "adapter_selection": "checkpoint-39185 selected previously on original MAGICDATA dev only; no selection on this test",
    "samples": int(metrics["samples"]),
    "cb_sensevoice_cer": baseline,
    "cb_sensevoice_plus_covo_cer": result,
    "absolute_cer_change": result - baseline,
    "relative_cer_change": ((result - baseline) / baseline) if baseline else None,
    "improved_samples": int(metrics["improved_samples"]),
    "worsened_samples": int(metrics["worsened_samples"]),
    "unchanged_samples": int(metrics["unchanged_samples"]),
    "artifacts": {
        "evidence": str((out / "cb_sensevoice_test.evidence.jsonl").resolve()),
        "messages": str((out / "cb_sensevoice_test.messages.jsonl").resolve()),
        "predictions": str((out / "checkpoint-39185.predictions.jsonl").resolve()),
        "summary": str(Path(sys.argv[2]).resolve()),
    },
}
(out / "final_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

touch "$OUT/ALL_DONE"
cat "$OUT/final_summary.json"
echo "[$(date -Is)] ALL_DONE"
