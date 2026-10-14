#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/root/autodl-tmp
SRC="$WORKSPACE/src"
PY="$WORKSPACE/great/bin/python"
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
CURRENT_OUT="$COVO/outputs/qwen35_9b_magicdata_cbsense_oraclelex6000_test_20260825"
OUT="$COVO/outputs/thchs30_zero_training_9b_suite_20260825"

PLAIN_INPUT="$SRC/logs/thchs30_nbest10_original_covo_messages_ckpt20000_20260811.jsonl"
CBSENSE_EVIDENCE_GZ="$SRC/logs/cb_sensevoice_thchs30_error_hotwords_evidence_20260811.jsonl.gz"
CBSENSE_EVIDENCE="$OUT/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"
CBSENSE_INPUT="$OUT/cb_sensevoice_thchs30_generic_selector.messages.jsonl"

declare -A ADAPTERS
ADAPTERS[magicdata]="$COVO/outputs/qwen35_9b_magicdata_hardneg_balanced_1epoch_20260820/checkpoint-39185"
ADAPTERS[stcmds]="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
ADAPTERS[aishell]="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
MODEL_ORDER=(magicdata stcmds aishell)

mkdir -p "$OUT/standalone" "$OUT/cbsense"

echo "[$(date -Is)] QUEUED waiting_for=magicdata_cbsense9b"
while pgrep -f '/run_magicdata_cbsense_qwen9b_20260825.sh' >/dev/null || \
      pgrep -f 'run_CLI.py test --config configs/cb-sensevoice-magicdata.yaml' >/dev/null || \
      pgrep -f 'infer_lora_text.py.*qwen35_9b_magicdata_cbsense_oraclelex6000_test_20260825' >/dev/null; do
  echo "[$(date -Is)] WAIT current_magicdata_combo_still_running"
  sleep 300
done

if [[ ! -e "$CURRENT_OUT/ALL_DONE" ]]; then
  echo "[error] preceding MAGICDATA combination job ended without ALL_DONE" >&2
  exit 1
fi

echo "[$(date -Is)] START no_training=true samples=2495"

for path in "$PY" "$MODEL" "$PLAIN_INPUT" "$CBSENSE_EVIDENCE_GZ"; do
  [[ -e "$path" ]] || { echo "[error] missing required path: $path" >&2; exit 2; }
done
for name in "${MODEL_ORDER[@]}"; do
  [[ -d "${ADAPTERS[$name]}" ]] || { echo "[error] missing adapter: ${ADAPTERS[$name]}" >&2; exit 2; }
done
[[ "$(wc -l < "$PLAIN_INPUT")" -eq 2495 ]] || { echo "[error] standalone input row count mismatch" >&2; exit 2; }

if [[ ! -s "$CBSENSE_EVIDENCE" ]]; then
  gzip -cd "$CBSENSE_EVIDENCE_GZ" > "$CBSENSE_EVIDENCE.inprogress"
  mv -f "$CBSENSE_EVIDENCE.inprogress" "$CBSENSE_EVIDENCE"
fi
[[ "$(wc -l < "$CBSENSE_EVIDENCE")" -eq 2495 ]] || { echo "[error] CB-Sense evidence row count mismatch" >&2; exit 2; }

if [[ ! -s "$CBSENSE_INPUT" ]]; then
  "$PY" "$SRC/analysis/cbsensevoice_covo_bridge.py" prepare \
    --input "$CBSENSE_EVIDENCE" --output "$CBSENSE_INPUT.inprogress" \
    --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 \
    --hotword-source prompt --include-pinyin --prompt-mode selector \
    --protect-supported-hotwords --clean-nbest --include-consensus-spans \
    --include-hotword-evidence --prefer-same-length --trust-asr-top1 \
    --preserve-anchor-digits
  mv -f "$CBSENSE_INPUT.inprogress" "$CBSENSE_INPUT"
fi
[[ "$(wc -l < "$CBSENSE_INPUT")" -eq 2495 ]] || { echo "[error] CB-Sense message row count mismatch" >&2; exit 2; }

"$PY" - "$OUT/run_manifest.json" "$MODEL" "$PLAIN_INPUT" "$CBSENSE_INPUT" \
  "${ADAPTERS[magicdata]}" "${ADAPTERS[stcmds]}" "${ADAPTERS[aishell]}" <<'PY'
import datetime
import json
import sys
from pathlib import Path

output, model, plain, cbsense, magicdata, stcmds, aishell = sys.argv[1:]
payload = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "dataset": "THCHS-30 full test",
    "samples": 2495,
    "training_performed": False,
    "selection_policy": "Adapters were locked using non-THCHS development/evaluation results; THCHS-30 is not used to select checkpoints or tune prompts.",
    "base_model": str(Path(model).resolve()),
    "routes": {
        "standalone_9b": {"classification": "full_cross_domain_no_training", "input": str(Path(plain).resolve())},
        "cbsense_plus_9b": {
            "classification": "leaked_diagnostic",
            "leakage_reason": "CB-SenseVoice error-targeted hotwords were derived from THCHS-30 test references/errors.",
            "input": str(Path(cbsense).resolve()),
        },
    },
    "adapters": {
        "magicdata": {"path": str(Path(magicdata).resolve()), "locked_checkpoint": 39185},
        "stcmds": {"path": str(Path(stcmds).resolve()), "locked_checkpoint": 34400},
        "aishell": {"path": str(Path(aishell).resolve()), "locked_checkpoint": 30022},
    },
    "generation": {"batch_size": 8, "max_new_tokens": 96, "temperature": 0.0, "disable_thinking": True},
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

run_eval() {
  local route="$1"
  local name="$2"
  local input="$3"
  local adapter="${ADAPTERS[$name]}"
  local pred="$OUT/$route/${name}.predictions.jsonl"
  local summary="$OUT/$route/${name}.summary.json"

  if [[ -s "$summary" ]]; then
    echo "[$(date -Is)] SKIP route=$route model=$name"
    return
  fi
  if [[ ! -s "$pred" ]]; then
    echo "[$(date -Is)] INFER_START route=$route model=$name rows=2495"
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" \
    "$PY" "$COVO/scripts/infer_lora_text.py" \
      --input "$input" --output "$pred.inprogress" \
      --model-name-or-path "$MODEL" --adapter-path "$adapter" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking
    mv -f "$pred.inprogress" "$pred"
  fi
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/evaluate_correction_jsonl.py" \
    --input "$pred" > "$summary"
  cat "$summary"
  echo "[$(date -Is)] INFER_DONE route=$route model=$name"
}

for name in "${MODEL_ORDER[@]}"; do
  run_eval standalone "$name" "$PLAIN_INPUT"
done
for name in "${MODEL_ORDER[@]}"; do
  run_eval cbsense "$name" "$CBSENSE_INPUT"
done

"$PY" - "$OUT" "$SRC/RESULTS_THCHS30_9B_20260825.md" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
doc_path = Path(sys.argv[2])
names = ["magicdata", "stcmds", "aishell"]

def read_route(route):
    return {name: json.loads((out / route / f"{name}.summary.json").read_text(encoding="utf-8")) for name in names}

standalone = read_route("standalone")
cbsense = read_route("cbsense")
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "dataset": "THCHS-30 full test",
    "samples": 2495,
    "training_performed": False,
    "selection_policy": "All adapters were locked outside THCHS-30; no THCHS-30 checkpoint or prompt tuning.",
    "standalone_9b": {"classification": "full_cross_domain_no_training", "models": standalone},
    "cbsense_plus_9b": {
        "classification": "leaked_diagnostic",
        "leakage_reason": "Error-targeted CB-SenseVoice hotwords were derived from THCHS-30 test references/errors.",
        "models": cbsense,
    },
    "lowest_observed_standalone": min(names, key=lambda name: standalone[name]["cer"]),
    "lowest_observed_cbsense_diagnostic": min(names, key=lambda name: cbsense[name]["cer"]),
}
(out / "final_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

labels = {"magicdata": "MAGICDATA 9B ckpt-39185", "stcmds": "ST-CMDS 9B ckpt-34400", "aishell": "AISHELL 9B ckpt-30022"}
lines = [
    "# THCHS-30 zero-training 9B evaluation, 2026-08-25",
    "",
    "All adapters were selected outside THCHS-30. No THCHS-30 training or checkpoint selection was performed.",
    "",
    "## Standalone 9B on SenseVoice 10-best",
    "",
    "| Adapter | Baseline CER | Output CER | Improved | Worsened | Unchanged |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in names:
    m = standalone[name]
    lines.append(f"| {labels[name]} | {100*m['baseline_cer']:.4f}% | {100*m['cer']:.4f}% | {m['improved_samples']} | {m['worsened_samples']} | {m['unchanged_samples']} |")
lines += [
    "",
    "Classification: **Full cross-domain, no training**.",
    "",
    "## CB-SenseVoice + 9B",
    "",
    "| Adapter | CB-Sense CER | Output CER | Improved | Worsened | Unchanged |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in names:
    m = cbsense[name]
    lines.append(f"| {labels[name]} | {100*m['baseline_cer']:.4f}% | {100*m['cer']:.4f}% | {m['improved_samples']} | {m['worsened_samples']} | {m['unchanged_samples']} |")
lines += [
    "",
    "Classification: **Leaked/Diagnostic**. The CB-SenseVoice error-targeted hotword list was derived from THCHS-30 test references/errors; these numbers are an upper-bound diagnostic, not a held-out main result.",
    "",
    "Full predictions, per-run summaries, the run manifest, and `final_summary.json` are stored in:",
    "",
    f"`{out.resolve()}`",
    "",
]
rendered = "\n".join(lines)
(out / "RESULTS_THCHS30_9B_20260825.md").write_text(rendered, encoding="utf-8")
doc_path.write_text(rendered, encoding="utf-8")
PY

touch "$OUT/ALL_DONE"
cat "$OUT/final_summary.json"
echo "[$(date -Is)] ALL_DONE"
