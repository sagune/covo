#!/usr/bin/env python3
import gzip
import json
import os

files = [
    "/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_error_targeted_covo_messages_full_20260727.jsonl.gz",
    "/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_error_targeted_covo_predictions_full_20260727.jsonl.gz",
    "/root/autodl-tmp/src/logs/stcmds_standard3139_acoustic_listwise_messages_20260729.jsonl.gz",
    "/root/autodl-tmp/src/logs/stcmds_standard3139_acoustic_listwise_predictions_20260729.jsonl.gz",
]
for p in files:
    print("=" * 92)
    if not os.path.exists(p):
        print("MISSING", p)
        continue
    with gzip.open(p, "rt", encoding="utf-8") as f:
        r = json.loads(f.readline())
    inp = r.get("input") or {}
    print(os.path.basename(p))
    print("  top keys:", sorted(r.keys()))
    print("  input keys:", sorted(inp.keys()))
    print("  has prediction:", "prediction" in r, "| reference:", bool(r.get("reference")))
    print("  nbest len:", len(inp.get("nbest") or []),
          "| cands len:", len((inp.get("cbwhisper") or {}).get("candidates") or []))
    for k in ("hotwords", "prompt_hotwords"):
        v = inp.get(k)
        print("  %s len=%s %s" % (k, len(v) if v else 0, json.dumps(v[:2], ensure_ascii=False) if v else ""))
