#!/usr/bin/env python3
"""P2b: anatomy of the residual selection gap.

For rows where a strictly better candidate exists inside the preserving subset,
is the difference real content, or number/punctuation formatting that the CER
metric punishes? This decides whether a better reranker can ever capture it.
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
CN_NUM = "零一二三四五六七八九十百千万亿两"


def strip_punct_digits(s):
    s = re.sub(r"[0-9]", "", s)
    s = re.sub("[" + CN_NUM + "]", "", s)
    return s


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


total_chars = 0
gap_rows = 0
cls = collections.Counter()
cer_by_cls = collections.Counter()
asr_gap = []
examples = collections.defaultdict(list)

for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    if not ref:
        continue
    total_chars += len(ref)
    cs = [dict(text=norm(c["text"]), asr=num(c, "asr_score"), total=num(c, "total_score"))
          for c in (inp.get("cbwhisper") or {}).get("candidates") or [] if isinstance(c, dict) and c.get("text")]
    if not cs:
        continue
    top = norm(inp.get("asr_top1") or "")
    pres = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in top]
    keep = [c for c in cs if all(t in c["text"] for t in pres)] or cs

    chosen = max(keep, key=lambda c: c["total"])
    oracle = min(keep, key=lambda c: edit_distance(list(ref), list(c["text"])))
    dc = edit_distance(list(ref), list(chosen["text"]))
    do = edit_distance(list(ref), list(oracle["text"]))
    if do >= dc or chosen["text"] == oracle["text"]:
        continue
    gap_rows += 1
    cer_by_cls["gap_pp"] += (dc - do)
    if strip_punct_digits(chosen["text"]) == strip_punct_digits(oracle["text"]):
        k = "digit/punct formatting only"
    elif chosen["text"] == oracle["text"].replace("，", "").replace(",", ""):
        k = "punctuation only"
    else:
        k = "content difference"
    cls[k] += 1
    asr_gap.append(oracle["asr"] - chosen["asr"])
    if len(examples[k]) < 3:
        examples[k].append((ref, chosen["text"], oracle["text"]))

print("total reference chars: %d" % total_chars)
print("rows with a strictly better in-subset candidate: %d" % gap_rows)
print("recoverable edits in those rows: %d  (= %.4f pp of corpus CER)" % (
    cer_by_cls["gap_pp"], 100 * cer_by_cls["gap_pp"] / total_chars))
print()
print("%-28s %6s %10s" % ("class", "rows", "edits"))
tot = 0
for k, v in cls.most_common():
    print("%-28s %6d" % (k, v))
print()
print("acoustic gap (oracle - chosen, raw asr_score): mean %+.4f  median %+.4f  min %+.4f  max %+.4f" % (
    sum(asr_gap) / len(asr_gap), sorted(asr_gap)[len(asr_gap) // 2], min(asr_gap), max(asr_gap)))
print()
for k, exs in examples.items():
    print("--- %s ---" % k)
    for ref, ch, orc in exs:
        print("  ref   :", ref)
        print("  chosen:", ch)
        print("  oracle:", orc)
