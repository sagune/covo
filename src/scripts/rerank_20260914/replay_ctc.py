#!/usr/bin/env python3
"""P2c: does a length-normalised forced-CTC score fix the tie-break?

The decoder's own asr_score points the wrong way inside the preserving subset.
This replay tests an independent, unbiased acoustic channel instead.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
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


def num(c, k, d=0.0):
    try:
        return float(c.get(k, d) or d)
    except (TypeError, ValueError):
        return d


ctc = {}
for l in (R / "ctc_scores.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
    ctc[str(r["id"])] = {norm(c["text"]): num(c, "asr_score") for c in cs if c.get("text")}
print("rows with forced-CTC scores: %d" % len(ctc))

prep = []
missing = 0
for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    if not ref:
        continue
    uid = pos2utt[int(r["id"])]
    scores = ctc.get(uid)
    cs = []
    for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        t = norm(c["text"])
        cs.append(dict(text=t, ln=len(t), asr=num(c, "asr_score"), total=num(c, "total_score"),
                       exact=num(c, "exact_weighted_score"), hot=num(c, "hotword_score"),
                       ctc=(scores or {}).get(t)))
    if not cs:
        continue
    if scores is None or sum(1 for c in cs if c["ctc"] is not None) < len(cs):
        missing += 1
    top = norm(inp.get("asr_top1") or "")
    pres = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in top]
    keep = [c for c in cs if all(t in c["text"] for t in pres)] or cs
    keep = [c for c in keep if c["ctc"] is not None] or keep
    prep.append(dict(ref=ref, top=top, keep=keep, desig=[norm(k) for k in desig.get(uid, [])]))

TC = sum(len(p["ref"]) for p in prep)
MEN = sum(len(p["desig"]) for p in prep)
print("rows usable: %d | rows with incomplete forced-CTC coverage: %d" % (len(prep), missing))
print("mean preserving-subset size: %.2f" % (sum(len(p["keep"]) for p in prep) / len(prep)))
print()


def evaluate(label, fn):
    err = hits = 0
    for p in prep:
        t = fn(p)["text"]
        err += edit_distance(list(p["ref"]), list(t))
        for k in p["desig"]:
            hits += k in t
    print("%-44s CER %7.4f%% | recall %6.2f%% (%4d)" % (label, 100 * err / TC, 100 * hits / MEN, hits))


evaluate("current (argmax total_score)", lambda p: max(p["keep"], key=lambda c: c["total"]))
evaluate("decoder asr_score", lambda p: max(p["keep"], key=lambda c: c["asr"]))
evaluate("forced-CTC raw", lambda p: max(p["keep"], key=lambda c: c["ctc"]))
evaluate("forced-CTC / len", lambda p: max(p["keep"], key=lambda c: c["ctc"] / max(c["ln"], 1)))
evaluate("forced-CTC / sqrt(len)", lambda p: max(p["keep"], key=lambda c: c["ctc"] / (max(c["ln"], 1) ** 0.5)))
evaluate("forced-CTC + exact coverage", lambda p: max(p["keep"], key=lambda c: (c["exact"], c["ctc"])))
evaluate("forced-CTC then decoder asr", lambda p: max(p["keep"], key=lambda c: (c["ctc"], c["asr"])))
evaluate("ORACLE in subset", lambda p: min(p["keep"], key=lambda c: edit_distance(list(p["ref"]), list(c["text"]))))
