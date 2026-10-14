#!/usr/bin/env bash
# Does the ST-CMDS regression come from the *insertion-strength* half of the relaxed
# admission?  The six relaxed parameters split into
#   breadth   : kws_positive_threshold, kws_topk_per_group,
#               kws_max_prompt_keywords, prompt_max_injected_keywords
#   insertion : sensevoice_hotword_token_weight, sensevoice_hotword_completion_weight
# This decodes ST-CMDS held-out with the breadth half only.  If its top-1 CER no
# longer regresses while recall still rises, the fix is to ship breadth-only and
# treat the insertion weights as domain-specific opt-in.
#
# Waits for the GPU queue to drain, then (only if the front end passes) runs the
# end-to-end COVO arm as well.
set -uo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
LOG="$R/stcmds_breadth.log"
DATA_ROOT="../datasets/stcmds/cb_sensevoice_heldout"
D="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
ORIG="$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl"
BASE_CER=5.7921

for i in $(seq 1 900); do
  [[ -f "$R/STCMDS_ORIG_FIXED_E2E_DONE" ]] && break
  sleep 60
done
echo "[$(date -Is)] GPU drained, breadth-only relax start" >> "$LOG"

cd "$WS/src"
if [[ ! -s "$R/stcmds_breadth.jsonl" ]]; then
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$R/stcmds_breadth.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-stcmds.yaml \
    --data.init_args.test_info.root="$DATA_ROOT" \
    --data.init_args.num_workers=0 \
    --model.init_args.root="$DATA_ROOT/hotword" \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$R/stcmds_breadth_oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$R/stcmds_breadth_oracle_summary.csv" \
    --model.init_args.kws_positive_threshold=0.5 \
    --model.init_args.kws_topk_per_group=12 \
    --model.init_args.kws_max_prompt_keywords=40 \
    --model.init_args.prompt_max_injected_keywords=8 >> "$LOG" 2>&1
fi

n=$(wc -l < "$R/stcmds_breadth.jsonl" 2>/dev/null || echo 0)
echo "[$(date -Is)] breadth decode rows=$n (expect 5130)" >> "$LOG"
if [[ "$n" -ne 5130 ]]; then
  echo "[$(date -Is)] ABORT: wrong row count" >> "$LOG"
  touch "$R/STCMDS_BREADTH_FAILED"
  exit 1
fi

cd "$WS"
"$PY" .dsh_checks/compare_pool_generic.py --uttid "$D/uttid" --aligned "$D/aligned.txt" \
  --label "ST-CMDS held-out front-end (breadth-only relax)" \
  --evidence "original(7/29)=$ORIG" \
  --evidence "relax(breadth+insertion)=$R/stcmds_relax.jsonl" \
  --evidence "relax(breadth only)=$R/stcmds_breadth.jsonl" >> "$LOG" 2>&1

# forced-CTC over the breadth pool, then the fixed rerank
cat > "$R/build_stcmds_breadth_ctc_input.py" <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
D = Path("/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test")
pos2utt = [l.split()[0] for l in (D / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
rows = []
for l in (R / "stcmds_breadth.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    rid = str(r.get("id", ""))
    if not rid.isdigit():
        continue
    cands = [str(c["text"]).strip() for c in ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
             if isinstance(c, dict) and c.get("text")]
    if not cands:
        continue
    rows.append(dict(id=pos2utt[int(rid)], input=dict(nbest=cands[:24]), reference=r.get("reference"), split="test"))
(R / "stcmds_breadth_ctc_input.jsonl").write_text(
    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps({"rows": len(rows)}))
PY
"$PY" "$R/build_stcmds_breadth_ctc_input.py" >> "$LOG" 2>&1

if [[ ! -s "$R/stcmds_breadth_ctc_scores.jsonl" ]]; then
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/stcmds_breadth_ctc_input.jsonl" --manifest "$R/stcmds_manifest.jsonl" \
    --output "$R/stcmds_breadth_ctc_scores.jsonl" --max-nbest 24 --progress-every 500 >> "$LOG" 2>&1
fi

"$PY" "$WS/.dsh_checks/apply_rerank_v2.py" \
  --evidence "$R/stcmds_breadth.jsonl" --ctc-scores "$R/stcmds_breadth_ctc_scores.jsonl" \
  --uttid "$D/uttid" --aligned "$D/aligned.txt" \
  --label "ST-CMDS breadth-only relax + FIXED rerank" --out "$R/stcmds_breadth_fixed.jsonl" \
  > "$R/stcmds_breadth_front.txt" 2>&1
cat "$R/stcmds_breadth_front.txt" >> "$LOG"

# gate: only spend the ~1.7 h end-to-end if the front end no longer regresses
front_cer=$(grep -oE 'front-end CER    : [0-9.]+% -> [0-9.]+%' "$R/stcmds_breadth_front.txt" | grep -oE '[0-9.]+' | head -1)
echo "[$(date -Is)] breadth-only front-end identity CER = ${front_cer:-?}% (baseline $BASE_CER%)" >> "$LOG"
pass=$(awk -v a="${front_cer:-99}" -v b="$BASE_CER" 'BEGIN{print (a<=b+0.02)?"yes":"no"}')
echo "[$(date -Is)] gate -> $pass" >> "$LOG"

if [[ "$pass" == "yes" ]]; then
  COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
          --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
          --prompt-mode selector --protect-supported-hotwords --clean-nbest
          --include-consensus-spans --include-hotword-evidence)
  MSG="$R/e2eSTCMDSBREADTH.messages.jsonl"
  PRED="$R/e2eSTCMDSBREADTH.predictions.jsonl"
  cd "$WS"
  if [[ ! -s "$MSG" ]]; then
    "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_breadth_fixed.jsonl" \
      --output "$MSG.inprogress" "${COMMON[@]}" >> "$LOG" 2>&1
    mv -f "$MSG.inprogress" "$MSG"
  fi
  cd "$COVO"
  if [[ ! -s "$PRED" ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
      --input "$MSG" --output "$PRED.inprogress" \
      --model-name-or-path "$MODEL" --adapter-path "$ADA_ST" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
    mv -f "$PRED.inprogress" "$PRED"
  fi
  echo "=== ST-CMDS breadth-only relax + fixed rerank, end-to-end ===" >> "$LOG"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" --label "ST-CMDS breadth+fixed e2e" \
    --aligned "$D/aligned.txt" --uttid-file "$D/uttid" >> "$LOG" 2>&1
fi

touch "$R/STCMDS_BREADTH_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
