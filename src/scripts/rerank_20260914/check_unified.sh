#!/usr/bin/env bash
# Can train.unified.jsonl be merged into the training set?  Check provenance and
# overlap before trusting "dataset: None".
set -uo pipefail
PY=/root/autodl-tmp/great/bin/python
"$PY" - <<'PY'
import json, gzip
from pathlib import Path

def norm(x):
    import sys
    sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
    from covo.text import normalize_chinese_text as n
    return n(x or "")

uni = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908/train.unified.jsonl")
tr = Path("/root/autodl-tmp/.dsh_checks/rerank/train_aishell_evidence.jsonl")

def load(p, limit=20000):
    out = []
    with p.open(encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            if i >= limit: break
            if l.strip():
                out.append(json.loads(l))
    return out

U = load(uni)
print("train.unified.jsonl rows:", len(U))
print("  keys:", sorted(U[0].keys()))
from collections import Counter
print("  dataset values:", Counter(str(r.get("dataset")) for r in U).most_common(5))
print("  split values:", Counter(str(r.get("split")) for r in U).most_common(5))
print("  source values:", Counter(str(r.get("source")) for r in U).most_common(5))
print("  row0 ref:", str(U[0].get("reference"))[:40])
print("  row0 top1:", str((U[0].get("input") or {}).get("asr_top1"))[:40])
print("  n candidates row0:", len(((U[0].get("input") or {}).get("cbwhisper") or {}).get("candidates") or []))
ids = Counter(str(r.get("id")) for r in U)
print("  distinct ids:", len(ids), "of", len(U))

# overlap of references with the current training set and with the red-line sets
tr_refs = {norm(json.loads(l).get("reference")) for l in tr.read_text(encoding="utf-8").splitlines() if l.strip()}
u_refs = {norm(r.get("reference")) for r in U}
print("  reference overlap with current AISHELL train set:", len(u_refs & tr_refs))

for label, p in (("ST-CMDS", "/root/autodl-tmp/.dsh_checks/stcmds/stcmds_evidence.jsonl"),
                 ("THCHS-30", "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl")):
    ev = {norm(json.loads(l).get("reference")) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}
    print("  RED LINE: reference overlap with %s = %d" % (label, len(u_refs & ev)))
PY
