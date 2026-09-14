#!/usr/bin/env python3
"""Cost side of relaxing: spurious hotword insertions (FIR-style) and length drift."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


for tag, path in (("base", R / "dev_base.jsonl"), ("relax", R / "dev_relax.jsonl")):
    if not path.exists():
        continue
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    spurious = rows_with_spurious = 0
    neg_rows = neg_rows_with_hotword = 0
    len_in = len_out = 0
    n = 0
    for r in rows:
        uid = pos2utt[int(r["id"])]
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        n += 1
        len_in += len(ref)
        len_out += len(top)
        hw = set(texts_of(inp.get("hotwords")))
        bad = [h for h in hw if h in top and h not in ref]
        if bad:
            spurious += len(bad)
            rows_with_spurious += 1
        if not desig.get(uid):
            neg_rows += 1
            if any(h in top for h in hw):
                neg_rows_with_hotword += 1
    print("%-6s rows=%d | spurious hotword insertions: %d in %d rows (%.2f%% of rows)" % (
        tag, n, spurious, rows_with_spurious, 100 * rows_with_spurious / n))
    print("       FIR-style (utterances with no designated hotword that emit a lexicon hotword): %d/%d = %.2f%%" % (
        neg_rows_with_hotword, neg_rows, 100 * neg_rows_with_hotword / max(neg_rows, 1)))
    print("       length drift on top-1 vs reference: %.4f" % (len_out / max(len_in, 1)))
