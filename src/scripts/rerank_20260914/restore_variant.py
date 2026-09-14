#!/usr/bin/env python3
"""Can module 2 keep working while module 1's recall is not lost?

Policy under test: keep the corrector's rewrite everywhere, but restore the
input text over protected hotword spans (character level, not utterance level).
That is the minimal intervention which guarantees a recall floor without
discarding the corrector's other edits.

Two protected sets are compared:
  deployable   prompt_hotwords / hotwords from the evidence (reference-blind)
  annotation   designated hotword mentions from aligned.txt (upper bound)
"""
import difflib
import json
import sys

TAR = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted"
sys.path.insert(0, TAR + "/covo/src")
sys.path.insert(0, "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")

from covo.edits import validate_edits                                # noqa: E402
from covo.text import normalize_chinese_text as norm                 # noqa: E402
from covo.metrics import edit_distance                               # noqa: E402

BASE = "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908"
DATA = "/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword"

rows = [json.loads(s) for s in open(BASE + "/scheduled/high_lr3e-5/full_625.predictions.jsonl")]
desig = {}
for line in open(DATA + "/dev/aligned.txt", encoding="utf-8-sig"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))


def span_list(text, terms):
    z = []
    for t in terms:
        if len(t) < 2:
            continue
        s = 0
        while True:
            p = text.find(t, s)
            if p < 0:
                break
            z.append((p, p + len(t)))
            s = p + 1
    return z


def restore(inp, pred, zone):
    if not zone:
        return pred
    out = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, inp, pred, autojunk=False).get_opcodes():
        if tag != "equal" and any(i < z1 and j > z0 for z0, z1 in zone):
            out.append(inp[i:j])          # protected span -> keep the front-end text
        else:
            out.append(pred[k:l])
    return "".join(out)


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def pool_of(r):
    p = set(texts_of(r["input"].get("nbest")))
    p |= set(texts_of((r["input"].get("cbwhisper") or {}).get("candidates")))
    return p


TOTAL = sum(len(norm(r["reference"])) for r in rows)
ALL = sum(len(desig.get(r["id"], [])) for r in rows)


def report(name, out):
    chars = err = edited = 0
    hits = destroyed = 0
    for r in rows:
        ref, inp = norm(r["reference"]), norm(r["input"]["asr_top1"])
        t = out[r["id"]]
        chars += len(ref)
        err += edit_distance(list(ref), list(t))
        edited += t != inp
        for k in desig.get(r["id"], []):
            hits += k in t
            destroyed += (k in inp and k not in t)
    sub = [r for r in rows if norm(r["reference"]) not in pool_of(r)]
    c = sum(len(norm(r["reference"])) for r in sub)
    ie = sum(edit_distance(list(norm(r["reference"])), list(norm(r["input"]["asr_top1"]))) for r in sub)
    oe = sum(edit_distance(list(norm(r["reference"])), list(out[r["id"]])) for r in sub)
    imp = sum(edit_distance(list(norm(r["reference"])), list(out[r["id"]])) <
              edit_distance(list(norm(r["reference"])), list(norm(r["input"]["asr_top1"]))) for r in sub)
    wor = sum(edit_distance(list(norm(r["reference"])), list(out[r["id"]])) >
              edit_distance(list(norm(r["reference"])), list(norm(r["input"]["asr_top1"]))) for r in sub)
    print("%-34s CER %7.4f%% | recall %5.2f%% (%d/%d) | destroyed %3d | edited %4d | beyond-N-best %+.3f pp (imp %d wor %d)" % (
        name, 100 * err / chars, 100 * hits / ALL, hits, ALL, destroyed, edited,
        100 * (ie - oe) / TOTAL, imp, wor))


variants = {"input top-1": {}, "raw 9B": {}, "restore[deployable]": {}, "restore[annotation]": {}}
for r in rows:
    inp, pred = norm(r["input"]["asr_top1"]), norm(r["prediction"])
    variants["input top-1"][r["id"]] = inp
    variants["raw 9B"][r["id"]] = pred
    dep = []
    for t in texts_of(r["input"].get("prompt_hotwords")) or texts_of(r["input"].get("hotwords")):
        if t in inp and any(t in c for c in pool_of(r)):
            dep.append(t)
    variants["restore[deployable]"][r["id"]] = restore(inp, pred, span_list(inp, dep))
    des = [k for k in desig.get(r["id"], []) if k in inp]
    variants["restore[annotation]"][r["id"]] = restore(inp, pred, span_list(inp, des))

print("rows %d | designated mentions %d\n" % (len(rows), ALL))
for name in ("input top-1", "raw 9B", "restore[deployable]", "restore[annotation]"):
    report(name, variants[name])
