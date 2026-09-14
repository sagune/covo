#!/usr/bin/env python3
"""Is CB-SenseVoice's own output evaluated identically across the prepared files?

Compares the same 1334 dev utterances as carried by two different pipelines:
  A) src/logs/hotword_lora_9b_20260908/dev.unified.jsonl     (bridge v2 -> training/eval line)
  B) src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl    (raw evidence -> selector line)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

BASE = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]

unified = [json.loads(l) for l in (BASE / "dev.unified.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
evidence = [json.loads(l) for l in (BASE / "dev.evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print("rows: unified=%d evidence=%d" % (len(unified), len(evidence)))

u = {r["id"]: r for r in unified}
e = {}
for r in evidence:
    e[pos2utt[int(r["id"])]] = r

ids = [i for i in u if i in e]
print("matched ids:", len(ids))

ref_diff = asr_diff = nb_diff = 0
examples = []
for i in ids:
    nu, ne = norm(u[i]["reference"]), norm(e[i]["reference"])
    au = norm(u[i]["input"]["asr_top1"])
    ae = norm(e[i]["input"]["asr_top1"])
    if nu != ne:
        ref_diff += 1
        if len(examples) < 5:
            examples.append(("reference", i, u[i]["reference"], e[i]["reference"]))
    if au != ae:
        asr_diff += 1
        if len(examples) < 10:
            examples.append(("asr_top1", i, u[i]["input"]["asr_top1"], e[i]["input"]["asr_top1"]))
    if [norm(x) for x in (u[i]["input"].get("nbest") or [])] != [norm(x) for x in (e[i]["input"].get("nbest") or [])]:
        nb_diff += 1

print("rows whose normalised reference differs      : %d/%d" % (ref_diff, len(ids)))
print("rows whose normalised asr_top1 differs       : %d/%d" % (asr_diff, len(ids)))
print("rows whose nbest candidate list differs      : %d/%d" % (nb_diff, len(ids)))
print()
for kind, i, a, b in examples:
    print("[%s] %s" % (kind, i))
    print("   unified : %r" % a)
    print("   evidence: %r" % b)
