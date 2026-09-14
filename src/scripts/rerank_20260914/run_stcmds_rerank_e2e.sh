#!/usr/bin/env bash
# Addendum: ST-CMDS reranked variant (protect + forced-CTC ordering), for parity with
# the AISHELL / THCHS-30 arms. Runs after the primary relax-only end-to-end.
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
D="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
EV="$R/stcmds_relax.jsonl"
PRED="$R/stcmds_reranked.predictions.jsonl"
LOG="$R/stcmds_rerank_e2e.log"

# wait for the primary e2e to finish so we do not contend for the GPU
for i in $(seq 1 480); do
  [ -f "$R/TRANSFER_E2E_DONE" ] && break
  sleep 60
done
echo "[$(date -Is)] addendum start" >> "$LOG"

# 1) forced-CTC scores over the relaxed pool
cat > "$R/build_stcmds_ctc_input.py" <<'PY'
import json, sys
from pathlib import Path
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
D = Path("/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test")
pos2utt = [l.split()[0] for l in (D / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
rows = []
for l in (R / "stcmds_relax.jsonl").read_text(encoding="utf-8").splitlines():
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
(R / "stcmds_relax_ctc_input.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps({"rows": len(rows)}))
PY
"$PY" "$R/build_stcmds_ctc_input.py" >> "$LOG" 2>&1

if [[ ! -s "$R/stcmds_relax_ctc_scores.jsonl" ]]; then
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$WS/src/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/stcmds_relax_ctc_input.jsonl" --manifest "$R/stcmds_manifest.jsonl" \
    --output "$R/stcmds_relax_ctc_scores.jsonl" --max-nbest 24 --progress-every 200 >> "$LOG" 2>&1
fi

"$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
  --evidence "$EV" --ctc-scores "$R/stcmds_relax_ctc_scores.jsonl" \
  --uttid "$D/uttid" --aligned "$D/aligned.txt" \
  --label "ST-CMDS relax + protect/forced-CTC rerank" \
  --out "$R/stcmds_relax_reranked.jsonl" >> "$LOG" 2>&1

# 2) bridge (no anchor) + infer
MSG="$R/e2eSTCMDS_RR.messages.jsonl"
cd "$WS"
if [[ ! -s "$MSG" ]]; then
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_relax_reranked.jsonl" \
    --output "$MSG.inprogress" --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 --hotword-source prompt \
    --include-pinyin --prompt-mode selector --protect-supported-hotwords --clean-nbest \
    --include-consensus-spans --include-hotword-evidence >> "$LOG" 2>&1
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

echo "=== ST-CMDS reranked end-to-end ===" >> "$LOG"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" --label "ST-CMDS reranked e2e" \
  --aligned "$D/aligned.txt" --uttid-file "$D/uttid" >> "$LOG" 2>&1

touch "$R/STCMDS_RR_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
