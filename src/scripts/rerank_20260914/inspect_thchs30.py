#!/usr/bin/env python3
import json
import os

P = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cbsense/aishell.predictions.jsonl"
with open(P) as f:
    r = json.loads(f.readline())
inp = r["input"]
print("top keys:", sorted(r.keys()))
print("input keys:", sorted(inp.keys()))
print("prediction:", (r.get("prediction") or "")[:70])
print("reference :", (r.get("reference") or "")[:70])
print("asr_top1  :", (inp.get("asr_top1") or "")[:70])
print("nbest len :", len(inp.get("nbest") or []))
print("cands len :", len((inp.get("cbwhisper") or {}).get("candidates") or []))
for k in ("hotwords", "prompt_hotwords", "covo_hotwords", "oracle_hotwords"):
    v = inp.get(k)
    print("%-16s len=%s %s" % (k, len(v) if v else 0, json.dumps(v[:3], ensure_ascii=False) if v else ""))
nc = inp.get("nbest_consensus") or {}
print("nbest_consensus keys:", sorted(nc.keys()) if isinstance(nc, dict) else type(nc))
print("  stable:", json.dumps(nc.get("stable_spans", [])[:2], ensure_ascii=False))
print("  uncertain:", json.dumps(nc.get("uncertain_spans", [])[:2], ensure_ascii=False))
km = inp.get("keyword_mentions")
print("keyword_mentions len=%s sample=%s" % (len(km) if km else 0,
      json.dumps(km[:2], ensure_ascii=False) if km else ""))

print()
print("=== THCHS-30 dataset dirs / possible hotword annotation ===")
for root in ("/root/autodl-tmp/datasets/thchs30",):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        print(" ", dirpath, "|", sorted(dirnames)[:6], "|", sorted(filenames)[:8])
