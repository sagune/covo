#!/usr/bin/env python3
"""P1c: do the existing forced-CTC scores cover the CB-SenseVoice candidate pool?"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

HL = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")

cb = {}
for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        cs = [norm(c["text"]) for c in (r["input"].get("cbwhisper") or {}).get("candidates") or [] if c.get("text")]
        cb[r["id"]] = dict(top=norm(r["input"].get("asr_top1") or ""), pool=set(cs), n=len(cs))

for name in ("acoustic_arbitration_20260913/scores.jsonl", "score_diagnostic_20260913/scores.jsonl", "acoustic_arbitration_20260913/input.jsonl"):
    p = HL / name
    if not p.exists():
        print("MISSING", p)
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts, same_top, overlap, sources = [], 0, [], {}
    for r in rows[:400]:
        cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
        texts = [norm(c.get("text") or "") for c in cs if c.get("text")]
        counts.append(len(texts))
        for c in cs:
            s = str(c.get("source") or "")
            sources[s] = sources.get(s, 0) + 1
        rid = r.get("id")
        if rid in cb and texts:
            same_top += texts[0] == cb[rid]["top"]
            overlap.append(len(set(texts) & cb[rid]["pool"]) / max(len(texts), 1))
    print("=== %s" % name)
    print("   rows=%d  candidates/row: mean %.2f median %.0f max %d" % (
        len(rows), statistics.mean(counts), statistics.median(counts), max(counts)))
    print("   sources: %s" % dict(sorted(sources.items(), key=lambda kv: -kv[1])[:5]))
    if overlap:
        print("   (sampled) candidates[0] == CB top-1: %d/%d | overlap with CB pool: mean %.2f" % (
            same_top, len(overlap), statistics.mean(overlap)))
    print()
