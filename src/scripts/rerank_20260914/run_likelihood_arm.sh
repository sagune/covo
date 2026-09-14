#!/usr/bin/env bash
# ============================================================================
# Likelihood-scoring arm: the same trained adapter used as a SCORER, not a generator.
#
# covo_candidate_likelihood_rerank.py computes, for each candidate the prompt actually
# showed, the mean log-probability of its {"text": ...} completion, and takes the argmax.
# No training, ~1 GPU-hour.
#
# What it is for
#   It separates two very different failures.  If the trained adapter can RANK the
#   candidates correctly but cannot GENERATE the right one, this arm jumps to the target
#   and we have learned that the bottleneck is decoding, not knowledge.  If the scorer is
#   also no better than the front end, then the 9B simply cannot see the choice, and no
#   amount of selection-side work (DPO, selector SFT) will help - which is worth knowing
#   before spending 3.5 h on DPO.
#
# What it is NOT
#   It cannot edit: its output is always a pool candidate, so on the 711 ST-CMDS rows
#   whose reference is not in the pool its beyond-N-best imp will be ~0 BY CONSTRUCTION.
#   Do not read that as "editing ability was lost" - it is a selector, and the plan's
#   editing dashboard applies to the generator.  It is a diagnostic and a control, and
#   the goal names it as such ("candidate-selector 对照").
#
# Verified before use: the numbered list in the prompt is exactly [asr_top1] + nbest
# (identical set AND order on all 5130 rows), so --max-candidates 9 scores precisely the
# candidates the model was shown.  --cb-weight stays 0 because input.cbwhisper's score
# field is the un-normalised total documented in the E38 audit.
# ============================================================================
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
CMP="$R/compare_old_vs_new.txt"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/likelihood_arm.log"
MSG="$R/e2eSTCMDS_VA.messages.jsonl"
PRED="$OUT/stcmds_likelihood.predictions.jsonl"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "============ LIKELIHOOD ARM START ============"
for i in $(seq 1 1400); do
  [ -f "$R/TRAIN_PLAN2_DONE" ] && break
  [ -f "$R/TRAIN_FAILED" ] && { say "training failed; aborting"; exit 1; }
  sleep 60
done
for i in $(seq 1 40); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [ -z "$busy" ] && break
  say "gpu busy ($busy), waiting"; sleep 60
done
[ -s "$OUT/final/adapter_model.safetensors" ] || { say "no trained adapter"; exit 1; }
[ -s "$MSG" ] || { say "missing messages: $MSG"; exit 1; }

if [ ! -s "$PRED" ]; then
  say "scoring START (trained adapter, 9 candidates, batch 8, max-length 2048)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$WS/src/analysis/covo_candidate_likelihood_rerank.py" \
    --input "$MSG" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$OUT/final" \
    --max-candidates 9 --batch-size 8 --max-length 2048 \
    --lm-weight 1.0 --cb-weight 0.0 --disable-thinking --progress-every 256 >> "$LOG" 2>&1
  rc=$?
  [ -s "$PRED.inprogress" ] && mv -f "$PRED.inprogress" "$PRED"
  say "scoring done rc=$rc rows=$(wc -l < "$PRED" 2>/dev/null || echo 0)"
  cd "$WS"
fi
[ -s "$PRED" ] || { say "no predictions produced"; exit 1; }

{
  echo
  echo "=== ST-CMDS: trained adapter as a SCORER (argmax over the shown candidates) ==="
  echo "    NOTE: a selector cannot edit, so its beyond-N-best imp is ~0 by construction."
} >> "$CMP"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" --label "ST-CMDS, trained adapter SCORER" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1
"$PY" "$WS/.dsh_checks/gain_split.py" --records "$PRED" --label "ST-CMDS trained-adapter SCORER" >> "$CMP" 2>&1
"$PY" "$WS/.dsh_checks/covo_error_anatomy.py" --records "$PRED" \
  --label "ST-CMDS trained-adapter SCORER" >> "$CMP" 2>&1

say "--- selected_rank distribution ---"
"$PY" - "$PRED" <<'PYEOF' >> "$LOG" 2>&1
import collections, json, sys
c = collections.Counter(); n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        r = json.loads(line); c[r.get("selected_rank")] += 1; n += 1
print("rows", n, "selected_rank:", dict(sorted(c.items(), key=lambda x: (x[0] is None, x[0]))))
print("stayed on rank 1 (copied top-1): %d (%.1f%%)" % (c.get(1, 0), 100.0 * c.get(1, 0) / max(1, n)))
print("deviated from top-1           : %d (%.1f%%)" % (n - c.get(1, 0), 100.0 * (n - c.get(1, 0)) / max(1, n)))
PYEOF

if [ -s "$R/e2eSTCMDS_VC.predictions.jsonl" ]; then
  "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$R/e2eSTCMDS_VC.predictions.jsonl" --b "$PRED" \
    --aligned "$S/aligned.txt" --uttid "$S/uttid" \
    --label-a "existing adapter V-C (generator)" --label-b "trained adapter SCORER" --draws 2000 >> "$CMP" 2>&1
fi

say "============ LIKELIHOOD ARM DONE ============"
touch "$R/LIKELIHOOD_DONE"
