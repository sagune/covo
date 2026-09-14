#!/usr/bin/env bash
# Follow-up: if the pool ceiling moved, re-score the new pools with forced-CTC and
# re-apply the rerank rule, so we can see whether the front-end reaches a better
# operating point on the enlarged pool.
set -euo pipefail

WS=/root/autodl-tmp
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
SCORER="$WS/src/analysis/score_sensevoice_candidate_evidence.py"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"

cat > "$R/build_pool_ctc_input.py" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
tag = sys.argv[1]
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
rows = []
for l in (R / ("dev_%s.jsonl" % tag)).read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    if not str(r.get("id", "")).isdigit():
        continue
    cands = [str(c["text"]).strip() for c in ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
             if isinstance(c, dict) and c.get("text")]
    if not cands:
        continue
    rows.append(dict(id=pos2utt[int(r["id"])], input=dict(nbest=cands[:24]), reference=r.get("reference"), split="dev"))
(R / ("%s_ctc_input.jsonl" % tag)).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps({"tag": tag, "rows": len(rows), "mean_candidates": sum(len(x["input"]["nbest"]) for x in rows) / max(len(rows), 1)}))
PY

MAN="$WS/src/logs/hotword_lora_9b_20260908/dual_ablation_20260909/audio_manifest.jsonl"
for tag in base relax; do
  [[ -s "$R/dev_$tag.jsonl" ]] || { echo "skip $tag (no evidence)"; continue; }
  "$PY" "$R/build_pool_ctc_input.py" "$tag"
  if [[ ! -s "$R/${tag}_ctc_scores.jsonl" ]]; then
    PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
      "$PY" "$SCORER" --input "$R/${tag}_ctc_input.jsonl" --manifest "$MAN" \
      --output "$R/${tag}_ctc_scores.jsonl" --max-nbest 24 --progress-every 200
  fi
  "$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
    --evidence "$R/dev_$tag.jsonl" --ctc-scores "$R/${tag}_ctc_scores.jsonl" \
    --uttid "$A/uttid" --aligned "$A/aligned.txt" \
    --label "pool-$tag (front-end, reranked)" --out "$R/dev_${tag}_reranked.jsonl"
done
touch "$R/POOL_RERANK_DONE"
echo "DONE"
