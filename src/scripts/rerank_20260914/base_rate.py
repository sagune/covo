#!/usr/bin/env python3
"""The base-rate mismatch that drives over-editing.

The training target is the reference, so every row teaches "move to the reference".  The
fraction of rows where that means CHANGING the input is the model's learned prior on how
often to edit.  If the training set's rate is much higher than a test set's, the model
enters that test set biased towards editing, and §10.12 shows precisely what that costs:
TCHS-30 and AISHELL dev already sit at 39-43% edit rates with 45-53% precision and
NEGATIVE net gain.

Reports two rates on both sides so the comparison is apples to apples:
  (a) reference != front-end top-1        -- the literal "did the target change the input"
  (b) some candidate beats top-1          -- the "was a better option visible" ceiling
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
OUT = R / "train_aishell_v1"
SETS = [
    ("TRAINING SET (AISHELL train, 17,301)", R / "train_aishell_evidence.jsonl"),
    ("ST-CMDS held-out", R / "e2eSTCMDS_VC.predictions.jsonl"),
    ("THCHS-30", R / "e2eTHCHS.predictions.jsonl"),
    ("AISHELL dev", R / "dev_reranked.predictions.jsonl"),
]


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


print("%-38s %8s %12s %12s" % ("set", "rows", "(a) ref!=top1", "(b) better exists"))
print("-" * 74)
for label, path in SETS:
    if not path.exists():
        print("%-38s MISSING %s" % (label, path))
        continue
    n = a_cnt = b_cnt = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        n += 1
        if ref != top:
            a_cnt += 1
        e0 = edit_distance(list(ref), list(top))
        vis = set(texts_of(inp.get("nbest")))
        if vis and min(edit_distance(list(ref), list(t)) for t in vis) < e0:
            b_cnt += 1
    print("%-38s %8d %11.1f%% %11.1f%%" % (label, n, 100.0 * a_cnt / n, 100.0 * b_cnt / n))

print()
print("The prior the model learns is (a) on the training set; the rate it meets at test")
print("time is (a) on that test set.  The ratio is the over-editing pressure.")
