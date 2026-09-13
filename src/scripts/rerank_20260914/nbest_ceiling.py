#!/usr/bin/env python3
"""What is the ceiling of the candidate set COVO actually sees?

Compares, on AISHELL dev:
  shipped top-1 / oracle over the model's visible N-best / oracle over the full pool
If the visible-N-best ceiling is well above the pool ceiling, the interface is
throwing away candidates and widening it is worth testing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

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


rows = []
for name in ("armB.messages.jsonl", "dev_reranked_selector.messages.jsonl"):
    p = R / name
    if p.exists():
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("using", name)
        break
if not rows:
    print("no messages file found")
    raise SystemExit(1)

TC = sum(len(norm(r.get("reference") or "")) for r in rows if norm(r.get("reference") or ""))
prep = []
sizes_nb, sizes_pool = [], []
for r in rows:
    uid = pos2utt[int(r["id"])]
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    top = norm(inp.get("asr_top1") or "")
    nb = texts_of(inp.get("nbest"))
    pool = texts_of((inp.get("cbwhisper") or {}).get("candidates"))
    if not ref or not nb:
        continue
    sizes_nb.append(len(nb))
    sizes_pool.append(len(pool))
    prep.append(dict(ref=ref, top=top, nb=nb, pool=pool or nb, desig=[norm(k) for k in desig.get(uid, [])]))

MEN = sum(len(p["desig"]) for p in prep)


def ev(name, fn):
    err = hits = 0
    for p in prep:
        t = fn(p)
        err += edit_distance(list(p["ref"]), list(t))
        for k in p["desig"]:
            hits += k in t
    print("  %-34s CER %7.4f%% | recall %6.2f%% (%d)" % (name, 100 * err / TC, 100 * hits / MEN, hits))


def best(p, cands):
    return min(cands, key=lambda t: edit_distance(list(p["ref"]), list(t)))


print("rows %d | visible N-best mean %.2f | pool mean %.2f" % (
    len(prep), sum(sizes_nb) / len(sizes_nb), sum(sizes_pool) / len(sizes_pool)))
print("reference inside visible N-best: %.1f%% | inside pool: %.1f%%" % (
    100 * sum(p["ref"] in p["nb"] for p in prep) / len(prep),
    100 * sum(p["ref"] in p["pool"] for p in prep) / len(prep)))
ev("shipped top-1", lambda p: p["top"])
ev("ORACLE over the visible N-best", lambda p: best(p, p["nb"]))
ev("ORACLE over the full pool", lambda p: best(p, p["pool"]))
