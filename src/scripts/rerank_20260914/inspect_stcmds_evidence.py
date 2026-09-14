#!/usr/bin/env python3
import gzip
import json
import os

F = "/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_standard3139_full_20260729.jsonl.gz"
print("size %.1f MB" % (os.path.getsize(F) / 1e6))
n = 0
with gzip.open(F, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        if n == 0:
            r = json.loads(line)
            inp = r.get("input") or {}
            print("top keys:", sorted(r.keys()))
            print("input keys:", sorted(inp.keys()))
            print("reference:", (r.get("reference") or "")[:60])
            print("asr_top1 :", (inp.get("asr_top1") or "")[:60])
            print("nbest len:", len(inp.get("nbest") or []),
                  "| cands len:", len((inp.get("cbwhisper") or {}).get("candidates") or []))
            cs = (inp.get("cbwhisper") or {}).get("candidates") or []
            print("candidate[0] keys:", sorted(cs[0].keys()) if cs else None)
            for k in ("hotwords", "prompt_hotwords"):
                v = inp.get(k)
                print("%-16s len=%s %s" % (k, len(v) if v else 0,
                      json.dumps(v[:3], ensure_ascii=False) if v else ""))
        n += 1
print("rows:", n)

print()
print("=== same check on other stcmds full evidence files ===")
for p in ["/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_error_targeted_full_20260727.jsonl.gz",
          "/root/autodl-tmp/src/logs/cb_sensevoice_stcmds_full_evidence_20260721.jsonl"]:
    if not os.path.exists(p):
        print("MISSING", p)
        continue
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        r = json.loads(f.readline())
    inp = r.get("input") or {}
    print("%-70s nbest=%s cands=%s hotwords=%s prompt_hw=%s" % (
        os.path.basename(p), len(inp.get("nbest") or []),
        len((inp.get("cbwhisper") or {}).get("candidates") or []),
        len(inp.get("hotwords") or []), len(inp.get("prompt_hotwords") or [])))
