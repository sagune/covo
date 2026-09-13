#!/usr/bin/env python3
"""Hotword-recognition headroom: what can a lexicon-driven correction layer add?

A ISHELL dev protocol: the designated hotword list is the task input. The
front-end already recognises most mentions. This measures the two directions:

  protect  : mentions present in the input top-1 that the corrector destroys
  recover  : mentions absent from the input top-1 that are still recoverable
             from the evidence pool (i.e. hotword recognition the correction
             layer could add rather than merely preserve)
"""
import json
import sys
from collections import Counter

TAR = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted"
sys.path.insert(0, TAR + "/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402

BASE = "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908"
DATA = "/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword"

print("=== hotword dir contents (the supplied list per split) ===")
import os  # noqa: E402
for split in ("dev", "test", "train"):
    d = os.path.join(DATA, split)
    if os.path.isdir(d):
        print(" ", split, sorted(os.listdir(d)))

rows = [json.loads(s) for s in open(BASE + "/scheduled/high_lr3e-5/full_625.predictions.jsonl")]

desig = {}
for line in open(DATA + "/dev/aligned.txt", encoding="utf-8-sig"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

# also count the lexicon itself if a flat word list exists
for cand in ("dev/keywords.txt", "dev/wordlist.txt", "dev/hotwords.txt", "dev/kws.txt"):
    p = os.path.join(DATA, cand)
    if os.path.exists(p):
        words = [norm(x) for x in open(p, encoding="utf-8-sig").read().split() if x.strip()]
        print("\nlexicon file %s: %d entries" % (cand, len(words)))

print("\n=== designated mentions: %d ===" % sum(len(v) for v in desig.values()))

in_input = in_pred = in_pool = missing = 0
missing_recoverable = missing_recovered = 0
destroyed = destroyed_recoverable = 0
per = Counter()
examples = []

for r in rows:
    inp = norm(r["input"]["asr_top1"])
    pred = norm(r["prediction"])
    pool = set()
    for x in r["input"].get("nbest") or []:
        pool.add(norm(x if isinstance(x, str) else x.get("text", "")))
    for c in (r["input"].get("cbwhisper") or {}).get("candidates") or []:
        if isinstance(c, dict) and c.get("text"):
            pool.add(norm(c["text"]))
    for k in desig.get(r["id"], []):
        a, b = k in inp, k in pred
        p = any(k in c for c in pool)
        in_input += a
        in_pred += b
        in_pool += p
        per[(a, b, p)] += 1
        if a and not b:
            destroyed += 1
            if p:
                destroyed_recoverable += 1
            if len(examples) < 4 and p:
                examples.append((r["id"], k, inp, pred))
        if not a:
            missing += 1
            if p:
                missing_recoverable += 1
            if b:
                missing_recovered += 1

print("present in input top-1            : %d" % in_input)
print("present in corrector output       : %d" % in_pred)
print("present somewhere in the evidence pool: %d" % in_pool)
print()
print("DESTROYED by the corrector        : %d  (of which the pool still holds it: %d)" % (
    destroyed, destroyed_recoverable))
print("MISSING from the input entirely   : %d  (of which the pool holds it: %d -> recoverable recognition)" % (
    missing, missing_recoverable))
print("  ...of those, the current corrector actually recovered: %d" % missing_recovered)
print()
print("=== (in_input, in_pred, in_pool) counts ===")
for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
    print("  input=%-5s output=%-5s pool=%-5s : %d" % (k[0], k[1], k[2], v))

if examples:
    print("\n=== destroyed but pool holds the correct form ===")
    for uid, k, inp, pred in examples:
        print("-" * 90)
        print("ID %s  hotword %s" % (uid, k))
        print("  IN :", inp)
        print("  OUT:", pred)
