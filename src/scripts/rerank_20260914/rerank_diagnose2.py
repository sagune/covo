#!/usr/bin/env python3
"""P1b: where exactly is the selection error?

Among candidates that already preserve the protected hotwords, how much better
could the tie-break be? This isolates "enforce preservation" (free) from
"choose well inside the preserving subset" (the real gap).
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

rows = [json.loads(l) for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def num(c, k, d=0.0):
    try:
        return float(c.get(k, d) or d)
    except (TypeError, ValueError):
        return d


prep = []
for r in rows:
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    if not ref:
        continue
    uid = pos2utt[int(r["id"])]
    cs = [dict(text=norm(c["text"]), asr=num(c, "asr_score"), search=num(c, "search_score"),
               hot=num(c, "hotword_score"), total=num(c, "total_score"), cons=num(c, "consensus_score"))
          for c in (inp.get("cbwhisper") or {}).get("candidates") or [] if isinstance(c, dict) and c.get("text")]
    if not cs:
        continue
    top = norm(inp.get("asr_top1") or "")
    pres = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in top]
    prep.append(dict(uid=uid, ref=ref, top=top, cs=cs, pres=pres, desig=[norm(k) for k in desig.get(uid, [])]))

TC = sum(len(p["ref"]) for p in prep)
MEN = sum(len(p["desig"]) for p in prep)


def evaluate(pickfn, label):
    err = hits = destroyed = 0
    for p in prep:
        t = pickfn(p)
        err += edit_distance(list(p["ref"]), list(t))
        for k in p["desig"]:
            hits += k in t
            destroyed += (k in p["top"] and k not in t)
    print("%-34s CER %7.4f%% | recall %6.2f%% (%4d) | destroyed %3d" % (
        label, 100 * err / TC, 100 * hits / MEN, hits, destroyed))


def within(p):  # preserving subset, fall back to all candidates
    keep = [c for c in p["cs"] if all(t in c["text"] for t in p["pres"])]
    return keep or p["cs"]


print("rows %d | mention %d" % (len(prep), MEN))
print()
print("=== full corpus ===")
evaluate(lambda p: p["top"], "current top-1")
evaluate(lambda p: max(p["cs"], key=lambda c: c["total"])["text"], "argmax total_score")
evaluate(lambda p: min(p["cs"], key=lambda c: edit_distance(list(p["ref"]), list(c["text"])))["text"], "pool oracle")

print()
print("=== inside the preserving subset (candidates that keep all protected hotwords) ===")
sizes = [len(within(p)) for p in prep]
print("subset size: mean %.2f  median %.0f  min %d  max %d  (rows with empty preserve set: %d)" % (
    statistics.mean(sizes), statistics.median(sizes), min(sizes), max(sizes),
    sum(1 for p in prep if not p["pres"])))
evaluate(lambda p: max(within(p), key=lambda c: c["total"])["text"], "argmax total_score")
evaluate(lambda p: max(within(p), key=lambda c: c["asr"])["text"], "argmax asr_score")
evaluate(lambda p: max(within(p), key=lambda c: c["search"])["text"], "argmax search_score")
evaluate(lambda p: max(within(p), key=lambda c: c["cons"])["text"], "argmax consensus")
evaluate(lambda p: min(within(p), key=lambda c: edit_distance(list(p["ref"]), list(c["text"])))["text"],
         "ORACLE within preserving subset")

print()
same = sum(1 for p in prep if max(within(p), key=lambda c: c["total"])["text"] == p["top"])
print("rows where current top-1 is also argmax total inside the subset: %d/%d" % (same, len(prep)))
best_in_sub = sum(1 for p in prep
                  if max(within(p), key=lambda c: c["total"])["text"] !=
                  min(within(p), key=lambda c: edit_distance(list(p["ref"]), list(c["text"])))["text"])
print("rows where a strictly better candidate exists inside the subset: %d/%d" % (best_in_sub, len(prep)))
