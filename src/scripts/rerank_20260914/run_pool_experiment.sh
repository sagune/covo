#!/usr/bin/env bash
# Round 3: can the candidate pool ceiling be raised?
#   base  : config defaults on the dev split (should reproduce dev.evidence.jsonl)
#   relax : looser KWS admission + stronger hotword bonus + more prompt injections
set -euo pipefail

WS=/root/autodl-tmp
SRC="$WS/src"
PY="$WS/great/bin/python"
OUT="$WS/.dsh_checks/rerank"
LOG="$OUT/pool.log"
mkdir -p "$OUT"

cd "$SRC"

run_one() {
  local tag="$1"; shift
  if [[ -s "$OUT/dev_${tag}.jsonl" ]]; then
    echo "[$(date -Is)] $tag already present ($(wc -l < "$OUT/dev_${tag}.jsonl") rows)" >> "$LOG"
    return
  fi
  echo "[$(date -Is)] $tag START" >> "$LOG"
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$OUT/dev_${tag}.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-aishell.yaml \
    --data.init_args.test_split=dev \
    --data.init_args.num_workers=0 \
    --model.init_args.split=dev \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$OUT/oracle_detail_${tag}.csv" \
    --model.init_args.oracle_nbest_summary_path="$OUT/oracle_summary_${tag}.csv" \
    "$@" >> "$LOG" 2>&1
  echo "[$(date -Is)] $tag DONE rows=$(wc -l < "$OUT/dev_${tag}.jsonl" 2>/dev/null || echo 0)" >> "$LOG"
}

run_one base
run_one relax \
  --model.init_args.kws_positive_threshold=0.5 \
  --model.init_args.kws_topk_per_group=12 \
  --model.init_args.kws_max_prompt_keywords=40 \
  --model.init_args.prompt_max_injected_keywords=8 \
  --model.init_args.sensevoice_hotword_token_weight=2.0 \
  --model.init_args.sensevoice_hotword_completion_weight=1.2

touch "$OUT/POOL_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
