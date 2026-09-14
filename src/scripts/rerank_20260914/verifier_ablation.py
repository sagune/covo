#!/usr/bin/env python3
"""Offline ablation of the plan's verifier on the cached CB-SenseVoice dev evidence.

Takes the existing full-sentence 9B predictions, diffs them against the input
top-1 to recover edit spans, then replays them through `covo.edits.validate_edits`
exactly as the plan specifies. No GPU, no retraining.

Variants:
  input            CB-SenseVoice top-1
  raw 9B           current deployed behaviour (full-sentence rewrite, no verifier)
  verifier         plan's verifier as implemented (from/to support, char cap)
  verifier+nogo    same, plus a no-edit zone over evidence-supported hotword spans
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
    out = []
    for c in (r["input"].get("cbwhisper") or {}).get("candidates") or []:
        if isinstance(c, dict) and c.get("text"):
            out.append(norm(c["text"]))
    return out


def diff_edits(a, b):
    """Recover (span in a) -> (span in b) edits, keeping positions for the no-go zone."""
    out = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((Edit(from_text=a[i:j], to_text=b[k:l]), i, j))
    return out


def evaluate(name, texts):
    chars = err = hits = tot = 0
    destroyed = 0
    for r, t in texts:
        ref, inp = norm(r["reference"]), norm(r["input"]["asr_top1"])
        chars += len(ref)
        err += edit_distance(list(ref), list(t))
        for k in desig.get(r["id"], []):
            tot += 1
            hits += k in t
            if k in inp and k not in t:
                destroyed += 1
    print("%-18s CER %7.4f%% | designated recall %5.2f%% (%d/%d) | destroyed %3d" % (
        name, 100 * err / chars, 100 * hits / tot, hits, tot, destroyed))


print("rows: %d" % len(rows))

# how many hotword spans qualify for the no-go zone
zone_rows = 0
for r in rows:
    inp = norm(r["input"]["asr_top1"])
    pool = set(nbest_of(r)) | set(cand_of(r))
    for k in desig.get(r["id"], []):
        if k in inp and any(k in c for c in pool):
            zone_rows += 1
            break
print("rows with at least one evidence-supported protected hotword span: %d (%.1f%%)" % (
    zone_rows, 100 * zone_rows / len(rows)))
print()

texts_input, texts_raw = [], []
recovered = {cap: [] for cap in (4, 8, 12, 16, 24, 10 ** 9)}
recovered_nogo = {cap: [] for cap in (4, 8, 12, 16, 24, 10 ** 9)}
verifier_changed = 0

for r in rows:
    inp = norm(r["input"]["asr_top1"])
    pred = norm(r["prediction"])
    texts_input.append((r, inp))
    texts_raw.append((r, pred))
    edits = diff_edits(inp, pred)
    evidence = {
        "asr_top1": inp,
        "nbest": nbest_of(r),
        "hotwords": [{"text": h["text"]} for h in r["input"].get("hotwords", [])],
    }
    # evidence-supported protected spans (no-edit zone)
    pool = set(nbest_of(r)) | set(cand_of(r))
    zone = []
    for k in desig.get(r["id"], []):
        if k in inp and any(k in c for c in pool):
            start = 0
            while True:
                p = inp.find(k, start)
                if p < 0:
                    break
                zone.append((p, p + len(k)))
                start = p + 1
    for cap in recovered:
        ok, _ = validate_edits([e for e, _, _ in edits], evidence,
                               max_total_changed_chars=cap, require_support=True)
        if ok:
            txt, _, rej = apply_edits(inp, [e for e, _, _ in edits])
            txt = txt if not rej else inp
            if cap == 12:
                verifier_changed += bool(txt != inp)
        else:
            txt = inp
        recovered[cap].append((r, txt))
        blocked = any(any(i < z1 and j > z0 for z0, z1 in zone) for _, i, j in edits) or not ok
        recovered_nogo[cap].append((r, inp if blocked else txt))

evaluate("input top-1", texts_input)
evaluate("raw 9B (deployed)", texts_raw)
print()
for cap in (4, 8, 12, 16, 24, 10 ** 9):
    label = "%d" % cap if cap < 10 ** 9 else "inf"
    evaluate("verifier cap=%s" % label, recovered[cap])
print()
for cap in (4, 8, 12, 16, 24, 10 ** 9):
    label = "%d" % cap if cap < 10 ** 9 else "inf"
    evaluate("verif+nogo cap=%s" % label, recovered_nogo[cap])
print()
print("rows the plan's verifier actually edited (cap=12): %d" % verifier_changed)
