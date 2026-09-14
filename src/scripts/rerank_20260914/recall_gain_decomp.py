#!/usr/bin/env python3
"""Decompose the designated-recall gain into three causes.

For every mention that the relaxed front-end now recalls but the base did not:
  newly retrieved  - the word was not in the base KWS list but is in relax's
  newly injected   - it was retrieved in both, but only relax injected it
  selection effect - same retrieved/injected evidence; the beam simply picked
                     a candidate containing it
"""
import argparse
import json
import sys
from collections import Counter
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


def load(path, pos2utt):
    out = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        rid = str(r.get("id", ""))
        uid = pos2utt[int(rid)] if rid.isdigit() and int(rid) < len(pos2utt) else rid
        inp = r.get("input") or {}
        out[uid] = dict(top=norm(inp.get("asr_top1") or ""),
                        hw=set(texts_of(inp.get("hotwords"))),
                        ph=set(texts_of(inp.get("prompt_hotwords"))))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--uttid", required=True)
ap.add_argument("--aligned", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--relax", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

B, R = load(a.base, pos2utt), load(a.relax, pos2utt)
c = Counter()
lost_kind = Counter()
for uid, kws in desig.items():
    if uid not in B or uid not in R:
        continue
    for k in kws:
        hb = k in B[uid]["top"]
        hr = k in R[uid]["top"]
        if hr and not hb:
            rb = any(k == h or k in h for h in B[uid]["hw"])
            rr = any(k == h or k in h for h in R[uid]["hw"])
            ib = any(k == h or k in h for h in B[uid]["ph"])
            ir = any(k == h or k in h for h in R[uid]["ph"])
            if not rb and rr:
                c["newly retrieved"] += 1
            elif rb and rr and not ib and ir:
                c["newly injected"] += 1
            else:
                c["selection effect (same admission)"] += 1
        elif hb and not hr:
            lost_kind["lost vs base"] += 1

gain = sum(c.values())
print("=== %s ===" % a.label)
print("designated mentions gained: %d | lost: %d" % (gain, sum(lost_kind.values())))
for k, v in c.most_common():
    print("  %-34s %4d  (%.1f%% of gains)" % (k, v, 100 * v / max(gain, 1)))
