#!/usr/bin/env python3
import json
import sys

paths = {
    "aishell_dev": "/root/autodl-tmp/src/logs/hotword_lora_9b_20260908/scheduled/high_lr3e-5/full_625.predictions.jsonl",
    "stcmds_standard3139": "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_9b_stcmds_cbsense_ablation_20260820/standard3139_chinesehp.predictions.jsonl",
    "thchs30_cbsense": "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cbsense/aishell.predictions.jsonl",
}

for name, p in paths.items():
    print("=" * 90)
    print(name, p)
    try:
        with open(p) as f:
            r = json.loads(f.readline())
    except Exception as e:
        print("  FAILED:", e)
        continue
    print("  top keys:", sorted(r.keys()))
    inp = r.get("input") or {}
    print("  input keys:", sorted(inp.keys()))
    print("  reference:", (r.get("reference") or "")[:60])
    print("  prediction:", (r.get("prediction") or "")[:60])
    print("  asr_top1:", (inp.get("asr_top1") or "")[:60])
    nb = inp.get("nbest")
    print("  nbest: type=%s len=%s first=%s" % (type(nb).__name__, len(nb) if nb else 0,
                                                (nb[0] if nb else None) if not isinstance(nb[0] if nb else None, dict) else sorted(nb[0].keys())))
    cw = inp.get("cbwhisper") or {}
    print("  cbwhisper keys:", sorted(cw.keys()) if isinstance(cw, dict) else type(cw))
    cs = cw.get("candidates") or []
    print("  candidates: len=%s first keys=%s" % (len(cs), sorted(cs[0].keys()) if cs else None))
    for k in ("hotwords", "prompt_hotwords"):
        v = inp.get(k)
        print("  %s: len=%s sample=%s" % (k, len(v) if v else 0, json.dumps(v[:2], ensure_ascii=False) if v else None))
    for k in ("nbest_consensus", "covo_hotwords", "keyword_mentions"):
        print("  has %s: %s" % (k, k in inp))
