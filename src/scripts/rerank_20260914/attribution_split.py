#!/usr/bin/env python3
"""Split the gain into front-end-attributable vs corrector-attributable.

A revert-based verifier earns its CER by putting CB-SenseVoice's own text back,
which cannot support a "generative correction module" contribution. This measures
how much of the gain survives when the corrector must actually produce the text.

Variants over the cached dev evidence (no GPU):
  input                 CB-SenseVoice top-1
  raw 9B                deployed full-sentence rewrite
  verify(all-or-nothing) plan verifier; reject -> revert whole utterance
  verify+nogo revert     same + whole-utterance revert on protected-span touch
  verify+nogo drop       same, but drop only the offending edits and keep the rest
"""
import difflib
import json
import sys

TAR = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted"
sys.path.insert(0, TAR + "/covo/src")
sys.path.insert(0, "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")

from covo.edits import Edit, apply_edits, validate_edits   # noqa: E402
from covo.text import normalize_chinese_text as norm       # noqa: E402
from workflow import edit_distance                          # noqa: E402

BASE = "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908"
DATA = "/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword"

rows = [json.loads(s) for s in open(BASE + "/scheduled/high_lr3e-5/full_625.predictions.jsonl")]
desig = {}
for line in open(DATA + "/dev/aligned.txt", encoding="utf-8-sig"):
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))


def nbest_of(r):
    out = []
    for x in r["input"].get("nbest") or []:
        if isinstance(x, str):
            out.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            out.append(norm(x["text"]))
    return out


def cand_of(r):
    return [norm(c["text"]) for c in (r["input"].get("cbwhisper") or {}).get("candidates") or []
            if isinstance(c, dict) and c.get("text")]


def diff_edits(a, b):
    out = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((Edit(from_text=a[i:j], to_text=b[k:l]), i, j))
    return out


def zone_of(r, inp):
    pool = set(nbest_of(r)) | set(cand_of(r))
    z = []
    for k in desig.get(r["id"], []):
        if k in inp and any(k in c for c in pool):
            s = 0
            while True:
                p = inp.find(k, s)
                if p < 0:
                    break
                z.append((p, p + len(k)))
                s = p + 1
    return z


def overlaps(e, z):
    _, i, j = e
    return any(i < z1 and j > z0 for z0, z1 in z)


def report(name, texts, inputs):
    chars = err = 0
    hits = tot = destroyed = 0
    edited = edit_err = edit_in_err = edit_chars = 0
    for (r, t), inp in zip(texts, inputs):
        ref = norm(r["reference"])
        ref_chars = len(ref)
        chars += ref_chars
        d = edit_distance(list(ref), list(t))
        err += d
        for k in desig.get(r["id"], []):
            tot += 1
            hits += k in t
            if k in inp and k not in t:
                destroyed += 1
        if t != inp:
            edited += 1
            edit_err += d
            edit_in_err += edit_distance(list(ref), list(inp))
            edit_chars += ref_chars
    gain = edit_in_err - edit_err
    print("%-22s CER %7.4f%% | recall %5.2f%% | destroyed %3d | edited %4d (%4.1f%%) | corrector-side gain on edited rows %+7.2f pp" % (
        name, 100 * err / chars, 100 * hits / tot, destroyed,
        edited, 100 * edited / len(texts), 100 * gain / chars))


inputs, raw, v_revert, v_drop = [], [], [], []
for r in rows:
    inp = norm(r["input"]["asr_top1"])
    pred = norm(r["prediction"])
    inputs.append(inp)
    raw.append(pred)
    edits = diff_edits(inp, pred)
    evidence = {"asr_top1": inp, "nbest": nbest_of(r),
                "hotwords": [{"text": h["text"]} for h in r["input"].get("hotwords", [])]}
    zone = zone_of(r, inp)

    ok, _ = validate_edits([e for e, _, _ in edits], evidence, max_total_changed_chars=12)
    t_all, _, rej = apply_edits(inp, [e for e, _, _ in edits])
    t_all = t_all if (ok and not rej) else inp
    v_revert.append(t_all) if not zone else v_revert.append(
        inp if any(overlaps(e, zone) for e in edits) else t_all)

    keep = [e for e in edits if not overlaps(e, zone)]
    ok2, _ = validate_edits([e for e, _, _ in keep], evidence, max_total_changed_chars=12)
    t_keep, _, rej2 = apply_edits(inp, [e for e, _, _ in keep])
    v_drop.append(t_keep if (ok2 and not rej2) else inp)

print("rows: %d\n" % len(rows))
report("input top-1", list(zip(rows, inputs)), inputs)
report("raw 9B", list(zip(rows, raw)), inputs)
report("verify", list(zip(rows, v_revert)) if False else list(zip(rows, v_revert)), inputs)
report("verify+nogo revert", list(zip(rows, v_revert)), inputs)
report("verify+nogo drop-only", list(zip(rows, v_drop)), inputs)
