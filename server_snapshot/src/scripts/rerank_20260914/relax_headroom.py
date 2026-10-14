#!/usr/bin/env python3
"""Selection headroom after relaxing admission (AISHELL dev).

Compares, on the relaxed pool:
  shipped top-1 / reranked top-1 / preserving-subset oracle / full pool oracle
so we can see how much of the remaining error is still "selection" rather than
"generation".
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


base = {json.loads(l)["id"]: json.loads(l) for l in (R / "dev_base.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
relax = {json.loads(l)["id"]: json.loads(l) for l in (R / "dev_relax.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

TC = MEN = 0
rows = []
for k, r in relax.items():
    if k not in base:
        continue
    uid = pos2utt[int(k)]
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    top = norm(inp.get("asr_top1") or "")
    btop = norm((base[k].get("input") or {}).get("asr_top1") or "")
    pool = texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest"))
    if not ref or not pool:
        continue
    protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in top]
    sub = [c for c in pool if all(p in c for p in protect)] or pool
    TC += len(ref)
    MEN += len(desig.get(uid, []))
    rows.append(dict(ref=ref, top=top, btop=btop, pool=pool, sub=sub, desig=desig.get(uid, [])))


def ev(name, fn):
    err = hits = 0
    for p in rows:
        t = fn(p)
        err += edit_distance(list(p["ref"]), list(t))
        for k in p["desig"]:
            hits += k in t
    print("  %-34s CER %7.4f%% | recall %6.2f%% (%4d/%d)" % (name, 100 * err / TC, 100 * hits / MEN, hits, MEN))


print("AISHELL dev, %d rows, %d designated mentions" % (len(rows), MEN))
print("mean pool size %.2f | mean preserving-subset %.2f" % (
    sum(len(p["pool"]) for p in rows) / len(rows), sum(len(p["sub"]) for p in rows) / len(rows)))
ev("base (original admission) top-1", lambda p: p["btop"])
ev("relax top-1", lambda p: p["top"])
ev("ORACLE over the relaxed pool", lambda p: min(p["pool"], key=lambda c: edit_distance(list(p["ref"]), list(c))))
ev("ORACLE over preserving subset", lambda p: min(p["sub"], key=lambda c: edit_distance(list(p["ref"]), list(c))))
