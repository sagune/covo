#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

BASE = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
raw = {json.loads(l)["id"]: json.loads(l) for l in (BASE / "dev.evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
prep = {json.loads(l)["id"]: json.loads(l) for l in (SEL / "aishell_selector.messages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

print("rows where the NORMALISED asr_top1 still differs:")
for i in prep:
    a, b = raw[i]["input"].get("asr_top1") or "", prep[i]["input"].get("asr_top1") or ""
    if norm(a) != norm(b):
        print("  id=%s" % i)
        print("    evidence : %r" % a)
        print("    prepared : %r" % b)
        print("    norm a   : %r" % norm(a))
        print("    norm b   : %r" % norm(b))

# does the same rewriting happen to the reference?
print()
print("rows where the NORMALISED reference differs:")
n = 0
for i in prep:
    a, b = raw[i].get("reference") or "", prep[i].get("reference") or ""
    if norm(a) != norm(b):
        n += 1
        if n <= 3:
            print("  id=%s\n    evidence : %r\n    prepared : %r" % (i, a, b))
print("  total: %d" % n)

# quantify: what characters does the bridge substitute in asr_top1?
import collections
subs = collections.Counter()
for i in prep:
    a, b = raw[i]["input"].get("asr_top1") or "", prep[i]["input"].get("asr_top1") or ""
    if a != b and len(a) == len(b):
        for x, y in zip(a, b):
            if x != y:
                subs[(x, y)] += 1
print()
print("character substitutions introduced by the bridge into asr_top1:", dict(subs))
