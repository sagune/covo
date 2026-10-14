#!/usr/bin/env bash
# Round 4: relaxed-admission transfer check on THCHS-30 (front-end + rerank).
set -euo pipefail

WS=/root/autodl-tmp
SRC="$WS/src"
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
LOG="$R/thchs.log"
D="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
ORIG="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"

cd "$SRC"

if [[ ! -s "$R/thchs_relax.jsonl" ]]; then
  echo "[$(date -Is)] thchs relax START" >> "$LOG"
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$R/thchs_relax.jsonl" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-thchs30-error-hotwords.yaml \
    --data.init_args.num_workers=0 \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$R/thchs_oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$R/thchs_oracle_summary.csv" \
    --model.init_args.kws_positive_threshold=0.5 \
    --model.init_args.kws_topk_per_group=12 \
    --model.init_args.kws_max_prompt_keywords=40 \
    --model.init_args.prompt_max_injected_keywords=8 \
    --model.init_args.sensevoice_hotword_token_weight=2.0 \
    --model.init_args.sensevoice_hotword_completion_weight=1.2 >> "$LOG" 2>&1
  echo "[$(date -Is)] thchs relax DONE rows=$(wc -l < "$R/thchs_relax.jsonl" 2>/dev/null || echo 0)" >> "$LOG"
fi

# front-end comparison vs the on-record evidence
"$PY" "$WS/.dsh_checks/compare_pool_generic.py" \
  --uttid "$D/uttid" --aligned "$D/aligned.txt" --label "THCHS-30 front-end" \
  --evidence "original=$ORIG" --evidence "relax(+rerank params)=$R/thchs_relax.jsonl" >> "$LOG" 2>&1

# forced-CTC scoring + protect/rerank on the new pool
cat > "$R/build_thchs_ctc_input.py" <<'PY'
import json, sys
from pathlib import Path
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
D = Path("/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test")
pos2utt = [l.split()[0] for l in (D / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
rows = []
for l in (R / "thchs_relax.jsonl").read_text(encoding="utf-8").splitlines():
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
(R / "thchs_ctc_input.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps({"rows": len(rows)}))
PY

"$PY" "$R/build_thchs_ctc_input.py" >> "$LOG" 2>&1
if [[ ! -s "$R/thchs_ctc_scores.jsonl" ]]; then
  PYTHONPATH="$SRC" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$SRC/analysis/score_sensevoice_candidate_evidence.py" \
    --input "$R/thchs_ctc_input.jsonl" --manifest "$R/thchs30_manifest.jsonl" \
    --output "$R/thchs_ctc_scores.jsonl" --max-nbest 24 --progress-every 200 >> "$LOG" 2>&1
fi

"$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
  --evidence "$R/thchs_relax.jsonl" --ctc-scores "$R/thchs_ctc_scores.jsonl" \
  --uttid "$D/uttid" --aligned "$D/aligned.txt" \
  --label "THCHS-30 relax + protect/forced-CTC rerank" \
  --out "$R/thchs_relax_reranked.jsonl" >> "$LOG" 2>&1

touch "$R/THCHS_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
