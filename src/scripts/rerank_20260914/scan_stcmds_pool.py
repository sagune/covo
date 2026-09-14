#!/usr/bin/env python3
"""Distribution of pool / hotword evidence across the ST-CMDS standard-3139 file."""
import gzip
import json
import statistics
import sys

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

F = "/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_standard3139_full_20260729.jsonl.gz"


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


nc, nh, npw, uniq = [], [], [], []
ref_in = 0
n = 0
ge4 = 0
with gzip.open(F, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        nb = texts_of(inp.get("nbest"))
        cs = texts_of((inp.get("cbwhisper") or {}).get("candidates"))
        pool = set(nb) | set(cs)
        nc.append(len(cs))
        nh.append(len(texts_of(inp.get("hotwords"))))
        npw.append(len(texts_of(inp.get("prompt_hotwords"))))
        uniq.append(len(pool))
        if len(pool) >= 4:
            ge4 += 1
        if norm(r.get("reference") or "") in pool:
            ref_in += 1
        n += 1

print("rows: %d" % n)
for name, v in (("candidates", nc), ("unique pool", uniq), ("hotwords", nh), ("prompt_hotwords", npw)):
    print("  %-16s mean %6.2f  median %4.1f  p10 %4.1f  p90 %4.1f  max %4d" % (
        name, statistics.mean(v), statistics.median(v),
        sorted(v)[len(v) // 10], sorted(v)[9 * len(v) // 10], max(v)))
print("  rows with pool>=4: %d (%.1f%%)" % (ge4, 100 * ge4 / n))
print("  reference in pool: %d (%.1f%%)" % (ref_in, 100 * ref_in / n))
