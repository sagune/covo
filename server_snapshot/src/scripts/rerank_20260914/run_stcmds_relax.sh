#!/usr/bin/env bash
# Round 14 (corrected): relaxed-admission transfer check on the ST-CMDS HELD-OUT set.
# The config defaults point at datasets/stcmds/cb_sensevoice (10,260 rows, the old
# non-held-out test set); the official runner overrides both roots. This script does
# the same, and additionally asserts the resulting evidence length is 5,130.
set -euo pipefail

WS=/root/autodl-tmp
SRC="$WS/src"
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
LOG="$R/stcmds_relax.log"
DATA_ROOT="../datasets/stcmds/cb_sensevoice_heldout"
D="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
ORIG="$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl"

cd "$SRC"

if [[ ! -s "$R/stcmds_relax.jsonl" ]]; then
  echo "[$(date -Is)] stcmds-relax HELD-OUT START (root=$DATA_ROOT)" >> "$LOG"
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$R/stcmds_relax.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-stcmds.yaml \
    --data.init_args.test_info.root="$DATA_ROOT" \
    --data.init_args.num_workers=0 \
    --model.init_args.root="$DATA_ROOT/hotword" \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$R/stcmds_oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$R/stcmds_oracle_summary.csv" \
    --model.init_args.kws_positive_threshold=0.5 \
    --model.init_args.kws_topk_per_group=12 \
    --model.init_args.kws_max_prompt_keywords=40 \
    --model.init_args.prompt_max_injected_keywords=8 \
    --model.init_args.sensevoice_hotword_token_weight=2.0 \
    --model.init_args.sensevoice_hotword_completion_weight=1.2 >> "$LOG" 2>&1
  echo "[$(date -Is)] stcmds-relax DONE rows=$(wc -l < "$R/stcmds_relax.jsonl" 2>/dev/null || echo 0)" >> "$LOG"
fi

n=$(wc -l < "$R/stcmds_relax.jsonl" 2>/dev/null || echo 0)
if [[ "$n" -ne 5130 ]]; then
  echo "[$(date -Is)] ERROR expected 5130 held-out rows, got $n" >> "$LOG"
  touch "$R/STCMDS_ROOT_MISMATCH"
  exit 1
fi

"$PY" "$WS/.dsh_checks/compare_pool_generic.py" \
  --uttid "$D/uttid" --aligned "$D/aligned.txt" --label "ST-CMDS held-out front-end" \
  --evidence "original(2026-07-29)=$ORIG" --evidence "relax=$R/stcmds_relax.jsonl" >> "$LOG" 2>&1

touch "$R/STCMDS_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
