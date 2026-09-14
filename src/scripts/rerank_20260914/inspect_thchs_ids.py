#!/usr/bin/env python3
import json

P = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cbsense/aishell.predictions.jsonl"
ids = []
with open(P) as f:
    for n, line in enumerate(f):
        r = json.loads(line)
        ids.append(str(r.get("id")))
        if n >= 4:
            break
print("record ids sample:", ids)

A = "/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt"
print("\naligned.txt head:")
lines = open(A, encoding="utf-8-sig").read().splitlines()
for l in lines[:5]:
    print("  ", repr(l))
align_ids = {l.split("\t")[1].strip() for l in lines if len(l.split("\t")) >= 2}
print("aligned unique uttids:", len(align_ids), "sample:", sorted(align_ids)[:5])

U = "/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid"
T = "/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/text"
ul = open(U, encoding="utf-8-sig").read().splitlines()
tl = open(T, encoding="utf-8-sig").read().splitlines()
print("\nuttid head:", [repr(x) for x in ul[:3]])
print("text   head:", [repr(x) for x in tl[:3]])
print("uttid count:", len(ul), "text count:", len(tl))
