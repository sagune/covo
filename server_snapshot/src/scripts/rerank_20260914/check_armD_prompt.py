#!/usr/bin/env python3
"""Did the forced-CTC rescoring actually reach the prompt?"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

R = Path("/root/autodl-tmp/.dsh_checks/rerank")

for name in ("armB.messages.jsonl", "armD.messages.jsonl"):
    p = R / name
    if not p.exists():
        print("MISSING", name)
        continue
    vals = []
    lines_with = Counter()
    n = 0
    for l in p.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        u = r["messages"][1]["content"]
        for m in re.finditer(r"asr=(-?[0-9.]+)", u):
            vals.append(float(m.group(1)))
        for key in ("N-best with reliability", "Hotword evidence", "Stable spans", "ASR top-1"):
            if key in u:
                lines_with[key] += 1
        n += 1
    if not vals:
        print("%s: no asr= values" % name)
        continue
    small = sum(1 for v in vals if abs(v) < 1.0)
    big = len(vals) - small
    print("%-22s rows=%d asr= values=%d | |v|<1 (forced-CTC scale): %d (%.1f%%) | >=1: %d (%.1f%%)" % (
        name, n, len(vals), small, 100 * small / len(vals), big, 100 * big / len(vals)))
    print("   min %.4f  max %.4f  median %.4f" % (min(vals), max(vals), sorted(vals)[len(vals) // 2]))
    print("   blocks seen:", dict(lines_with))
