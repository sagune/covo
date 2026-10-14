#!/usr/bin/env python3
"""Does the bridge change the reference / asr_top1 that the evaluation uses?

Compares raw evidence against the bridge-prepared messages, which is what the
selector run actually evaluated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

BASE = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")

raw = {json.loads(l)["id"]: json.loads(l) for l in (BASE / "dev.evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
prep = {json.loads(l)["id"]: json.loads(l) for l in (SEL / "aishell_selector.messages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
print("rows: raw=%d prep=%d" % (len(raw), len(prep)))

ref_raw_ne = top_raw_ne = ref_norm_ne = top_norm_ne = 0
ex = []
for i in prep:
    a, b = raw[i], prep[i]
    if a.get("reference") != b.get("reference"):
        ref_raw_ne += 1
        if len(ex) < 6:
            ex.append(("reference", i, a.get("reference"), b.get("reference")))
    if (a["input"].get("asr_top1")) != (b["input"].get("asr_top1")):
        top_raw_ne += 1
        if len(ex) < 12:
            ex.append(("asr_top1", i, a["input"].get("asr_top1"), b["input"].get("asr_top1")))
    if norm(a.get("reference") or "") != norm(b.get("reference") or ""):
        ref_norm_ne += 1
    if norm(a["input"].get("asr_top1") or "") != norm(b["input"].get("asr_top1") or ""):
        top_norm_ne += 1

print("reference differs  (raw string) : %d/%d" % (ref_raw_ne, len(prep)))
print("asr_top1  differs  (raw string) : %d/%d" % (top_raw_ne, len(prep)))
print("reference differs  (normalised) : %d/%d" % (ref_norm_ne, len(prep)))
print("asr_top1  differs  (normalised) : %d/%d" % (top_norm_ne, len(prep)))
print()
for kind, i, a, b in ex:
    print("[%s] id=%s" % (kind, i))
    print("   evidence : %r" % a)
    print("   prepared : %r" % b)

# also: does the prepared file keep prompt_hotwords/hotwords (they matter for the protection set)?
missing_hw = sum(1 for i in prep if not (prep[i]["input"].get("prompt_hotwords") or prep[i]["input"].get("hotwords")))
print()
print("prepared rows without any hotword field: %d/%d" % (missing_hw, len(prep)))
