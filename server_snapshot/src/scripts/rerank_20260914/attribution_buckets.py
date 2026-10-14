#!/usr/bin/env python3
"""Proper attribution: is the final text reachable by the front-end, or generated?

Bucket every row's final text into:
  same-as-input        no correction happened
  in-pool selection    final text is one of the front-end's own candidates
  out-of-pool          final text exists nowhere in the front-end evidence
Only the third bucket is value that no reranker over this pool could ever get.
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


def overlap(e, z):
    _, i, j = e
    return any(i < z1 and j > z0 for z0, z1 in z)


variants = {}
inputs = {}
for r in rows:
    inp = norm(r["input"]["asr_top1"])
    inputs[r["id"]] = inp
    pool = set(nbest_of(r)) | set(cand_of(r))
    zone = []
    for k in desig.get(r["id"], []):
        if k in inp and any(k in c for c in pool):
            s = 0
            while True:
                p = inp.find(k, s)
                if p < 0:
                    break
                zone.append((p, p + len(k)))
                s = p + 1
    edits = diff_edits(inp, norm(r["prediction"]))
    ev = {"asr_top1": inp, "nbest": nbest_of(r),
          "hotwords": [{"text": h["text"]} for h in r["input"].get("hotwords", [])]}

    variants.setdefault("raw 9B", {})[r["id"]] = norm(r["prediction"])

    ok, _ = validate_edits([e for e, _, _ in edits], ev, max_total_changed_chars=12)
    t, _, rej = apply_edits(inp, [e for e, _, _ in edits])
    variants.setdefault("plan verifier", {})[r["id"]] = t if (ok and not rej) else inp

    keep = [e for e in edits if not overlap(e, zone)]
    ok2, _ = validate_edits([e for e, _, _ in keep], ev, max_total_changed_chars=12)
    t2, _, rej2 = apply_edits(inp, [e for e, _, _ in keep])
    variants.setdefault("verifier + no-edit zone", {})[r["id"]] = t2 if (ok2 and not rej2) else inp

total_chars = sum(len(norm(r["reference"])) for r in rows)
print("total reference chars: %d\n" % total_chars)

for name, out in variants.items():
    buckets = {"same-as-input": [], "in-pool selection": [], "out-of-pool generation": []}
    for r in rows:
        inp = inputs[r["id"]]
        t = out[r["id"]]
        pool = set(nbest_of(r)) | set(cand_of(r))
        key = "same-as-input" if t == inp else ("in-pool selection" if t in pool else "out-of-pool generation")
        buckets[key].append(r)
    print("### %s" % name)
    print("  %-26s %5s %10s %10s %10s %8s" % ("bucket", "rows", "in CER", "out CER", "gain pp", "recall"))
    for key, rs in buckets.items():
        if not rs:
            continue
        chars = sum(len(norm(r["reference"])) for r in rs)
        ie = sum(edit_distance(list(norm(r["reference"])), list(inputs[r["id"]])) for r in rs)
        oe = sum(edit_distance(list(norm(r["reference"])), list(out[r["id"]])) for r in rs)
        tot = sum(len(desig.get(r["id"], [])) for r in rs)
        hits = sum(sum(k in out[r["id"]] for k in desig.get(r["id"], [])) for r in rs)
        print("  %-26s %5d %9.3f%% %9.3f%% %+9.3f %7.1f%%" % (
            key, len(rs), 100 * ie / chars, 100 * oe / chars, 100 * (ie - oe) / total_chars,
            100 * hits / max(tot, 1)))
    ce = sum(edit_distance(list(norm(r["reference"])), list(out[r["id"]])) for r in rows)
    print("  TOTAL CER %.4f%%\n" % (100 * ce / total_chars))
