#!/usr/bin/env python3
"""Attribute the relax gain: retrieval side vs decode side.

For each designated mention, compare (base, relax) on two axes:
  retrieved  - present in the KWS-retrieved hotword list
  in_pool    - present in at least one candidate
This separates "the word was admitted" from "the word reached a hypothesis".
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
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        rid = str(r.get("id", ""))
        uid = pos2utt[int(rid)] if rid.isdigit() and int(rid) < len(pos2utt) else rid
        inp = r.get("input") or {}
        out[uid] = dict(
            pool=texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest")),
            hw=set(texts_of(inp.get("hotwords"))),
            ph=set(texts_of(inp.get("prompt_hotwords"))),
        )
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

B = load(a.base, pos2utt)
R = load(a.relax, pos2utt)
c = Counter()
for uid, kws in desig.items():
    if uid not in B or uid not in R:
        continue
    for k in kws:
        rb = any(k == h or k in h for h in B[uid]["hw"])
        rr = any(k == h or k in h for h in R[uid]["hw"])
        pb = any(k in x for x in B[uid]["pool"])
        pr = any(k in x for x in R[uid]["pool"])
        c[("retr", rb, rr)] += 1
        c[("pool", pb, pr)] += 1

print("=== %s ===" % a.label)
print("retrieved-set sizes: base mean %.2f | relax mean %.2f" % (
    sum(len(v["hw"]) for v in B.values()) / len(B), sum(len(v["hw"]) for v in R.values()) / len(R)))
print("prompt-injected sizes: base mean %.2f | relax mean %.2f" % (
    sum(len(v["ph"]) for v in B.values()) / len(B), sum(len(v["ph"]) for v in R.values()) / len(R)))
print()
tot = sum(v for (kind, _, _), v in c.items() if kind == "retr")
print("designated mentions tracked: %d" % tot)
print("  retrieved:  base %d -> relax %d  (+%d newly admitted)" % (
    sum(v for (k, x, y), v in c.items() if k == "retr" and x),
    sum(v for (k, x, y), v in c.items() if k == "retr" and y),
    c[("retr", False, True)]))
print("  in pool  :  base %d -> relax %d  (+%d newly in a candidate)" % (
    sum(v for (k, x, y), v in c.items() if k == "pool" and x),
    sum(v for (k, x, y), v in c.items() if k == "pool" and y),
    c[("pool", False, True)]))
print()
print("retrieval-side gains (newly admitted):      %d" % c[("retr", False, True)])
print("  ...of which also reached a candidate:     %d" % sum(
    1 for uid, kws in desig.items() if uid in B and uid in R
    for k in kws
    if (not any(k == h or k in h for h in B[uid]["hw"])) and any(k == h or k in h for h in R[uid]["hw"])
    and any(k in x for x in R[uid]["pool"])))
print("decode-side gains (was admitted, now in pool): %d" % sum(
    1 for uid, kws in desig.items() if uid in B and uid in R
    for k in kws
    if any(k == h or k in h for h in B[uid]["hw"]) and (not any(k in x for x in B[uid]["pool"]))
    and any(k in x for x in R[uid]["pool"])))
print("regressions (in base pool, not in relax pool): %d" % c[("pool", True, False)])
