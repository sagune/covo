#!/usr/bin/env python3
import json
import os

cands = [
    "/root/autodl-tmp/src/logs/stcmds_cb_sensevoice_standard3139_chinesehp_messages_full_20260729.jsonl",
    "/root/autodl-tmp/src/logs/stcmds_cb_sensevoice_standard3139_hardnegative_predictions_full_20260729.jsonl",
    "/root/autodl-tmp/src/logs/stcmds_cb_sensevoice_targeted_chinesehp_messages_20260728.jsonl",
]
for p in cands:
    print("=" * 90)
    print(p, "size=%.1fMB" % (os.path.getsize(p) / 1e6) if os.path.exists(p) else "MISSING")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        r = json.loads(f.readline())
    inp = r.get("input") or {}
    print("  top keys:", sorted(r.keys()))
    print("  input keys:", sorted(inp.keys()))
    print("  has prediction:", "prediction" in r, "| has reference:", bool(r.get("reference")))
    print("  nbest len:", len(inp.get("nbest") or []),
          "| cands len:", len((inp.get("cbwhisper") or {}).get("candidates") or []))
    for k in ("hotwords", "prompt_hotwords"):
        v = inp.get(k)
        print("  %s len=%s %s" % (k, len(v) if v else 0, json.dumps(v[:2], ensure_ascii=False) if v else ""))

print()
print("=== other stcmds evidence/messages files in src/logs ===")
for f in sorted(os.listdir("/root/autodl-tmp/src/logs")):
    if "stcmds" in f and ("evidence" in f or "messages" in f or "predictions" in f):
        s = os.path.getsize("/root/autodl-tmp/src/logs/" + f)
        print("  %8.1fMB  %s" % (s / 1e6, f))
