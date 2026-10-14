#!/usr/bin/env python3
"""Spurious-hotword-insertion cost for any evidence file (generic)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


ap = argparse.ArgumentParser()
ap.add_argument("--uttid", required=True)
ap.add_argument("--evidence", action="append", required=True, help="LABEL=PATH")
a = ap.parse_args()
pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]

for item in a.evidence:
    label, _, path = item.partition("=")
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print("%-26s (not present)" % label)
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    spur = rows_spur = n = 0
    len_ref = len_top = 0
    for r in rows:
        rid = str(r.get("id", ""))
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        n += 1
        len_ref += len(ref)
        len_top += len(top)
        hw = set(texts_of(inp.get("hotwords")))
        bad = [h for h in hw if h in top and h not in ref]
        spur += len(bad)
        rows_spur += bool(bad)
    print("%-26s rows=%5d | spurious hotword insertions %5d in %4d rows (%.2f%%) | length drift %.4f" % (
        label, n, spur, rows_spur, 100 * rows_spur / max(n, 1), len_top / max(len_ref, 1)))
