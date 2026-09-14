#!/usr/bin/env bash
# Measure prompt token lengths so the training-time estimate is grounded.
set -uo pipefail
PY=/root/autodl-tmp/great/bin/python
"$PY" - <<'PY'
import json, statistics as st
from pathlib import Path

TOK = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=False)
    ok = True
except Exception as e:
    print("tokenizer unavailable:", str(e)[:120]); ok = False

def lens_of(path, n=400, key="messages"):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            if i >= n: break
            if not l.strip(): continue
            r = json.loads(l)
            msgs = r.get(key) or []
            if not msgs: continue
            txt = "".join(m.get("content", "") for m in msgs)
            out.append(len(tok(txt).input_ids) if ok else len(txt))
    return out

for label, path in (
    ("our VD prompt (ST-CMDS)", "/root/autodl-tmp/.dsh_checks/rerank/e2eSTCMDS_VD.messages.jsonl"),
    ("existing training data (old format)", "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/stcmds_chinesehp_sensevoice/train.cb-hardnegative.qwen.jsonl"),
):
    L = lens_of(path)
    if not L:
        print("%-38s (not found)" % label); continue
    L.sort()
    print("%-38s n=%d  min=%d  p50=%d  p90=%d  p99=%d  max=%d  mean=%.0f" % (
        label, len(L), L[0], L[len(L)//2], L[int(0.9*len(L))], L[int(0.99*len(L))], L[-1], st.mean(L)))
    print("%-38s  %% > 1024 tokens: %.1f%%   %% > 1536: %.1f%%   %% > 2048: %.1f%%" % (
        "", 100*sum(1 for x in L if x > 1024)/len(L),
        100*sum(1 for x in L if x > 1536)/len(L),
        100*sum(1 for x in L if x > 2048)/len(L)))
PY
