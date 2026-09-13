#!/usr/bin/env python3
"""P2 (first pass): can a better tie-break be built from the EXISTING candidate fields?

All rules pick inside the preserving subset (candidates that keep every protected
hotword), so the comparison isolates the tie-break, holding the constraint fixed.
"""
import json
import math
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
for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    if not ref:
        continue
    uid = pos2utt[int(r["id"])]
    cs = []
    for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        t = norm(c["text"])
        cs.append(dict(text=t, ln=len(t), asr=num(c, "asr_score"), search=num(c, "search_score"),
                       hot=num(c, "hotword_score"), exact=num(c, "exact_weighted_score"),
                       cons=num(c, "consensus_score"), sup=num(c, "consensus_support"),
                       total=num(c, "total_score")))
    if not cs:
        continue
    top = norm(inp.get("asr_top1") or "")
    pres = [x for x in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if x in top]

    def within(cands):
        k = [c for c in cands if all(x in c["text"] for x in pres)]
        return k or cands

    keep = within(cs)
    # medoid: minimise total edit distance to the other candidates in the subset
    for a in keep:
        a["medoid"] = sum(edit_distance(list(a["text"]), list(b["text"])) for b in keep if b is not a)
    # character-level agreement: how many other candidates share each of this candidate's chars at the same index
    for a in keep:
        a["agree"] = sum(1 for b in keep if b is not a for i, ch in enumerate(a["text"]) if i < len(b["text"]) and b["text"][i] == ch)
    prep.append(dict(ref=ref, top=top, keep=keep, desig=[norm(k) for k in desig.get(uid, [])]))

TC = sum(len(p["ref"]) for p in prep)
MEN = sum(len(p["desig"]) for p in prep)
print("rows %d | mention %d | mean preserving-subset size %.2f" % (
    len(prep), MEN, sum(len(p["keep"]) for p in prep) / len(prep)))


def evaluate(label, fn):
    err = hits = 0
    for p in prep:
        t = fn(p)["text"]
        err += edit_distance(list(p["ref"]), list(t))
        for k in p["desig"]:
            hits += k in t
    print("%-40s CER %7.4f%% | recall %6.2f%% (%4d)" % (label, 100 * err / TC, 100 * hits / MEN, hits))


evaluate("current (argmax total on full pool: reference)",
         lambda p: max(p["keep"], key=lambda c: c["total"]))
evaluate("asr raw", lambda p: max(p["keep"], key=lambda c: c["asr"]))
evaluate("asr / len", lambda p: max(p["keep"], key=lambda c: c["asr"] / max(c["ln"], 1)))
evaluate("asr / sqrt(len)", lambda p: max(p["keep"], key=lambda c: c["asr"] / math.sqrt(max(c["ln"], 1))))
for lam in (0.5, 1.0, 2.0, 4.0):
    evaluate("asr - %.1f*len" % lam, (lambda L: (lambda p: max(p["keep"], key=lambda c: c["asr"] - L * c["ln"])))(lam))
evaluate("search raw", lambda p: max(p["keep"], key=lambda c: c["search"]))
evaluate("consensus support (raw count)", lambda p: max(p["keep"], key=lambda c: (c["sup"], c["asr"])))
evaluate("medoid (min sum edit-distance to peers)", lambda p: min(p["keep"], key=lambda c: c["medoid"]))
evaluate("medoid, tie-break by asr", lambda p: min(p["keep"], key=lambda c: (c["medoid"], -c["asr"])))
evaluate("char-agreement then asr", lambda p: max(p["keep"], key=lambda c: (c["agree"], c["asr"])))
evaluate("exact coverage then asr", lambda p: max(p["keep"], key=lambda c: (c["exact"], c["asr"])))
evaluate("ORACLE in subset", lambda p: min(p["keep"], key=lambda c: edit_distance(list(p["ref"]), list(c["text"]))))
