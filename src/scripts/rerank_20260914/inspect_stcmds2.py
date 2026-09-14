#!/usr/bin/env python3
import glob
import gzip
import json
import os

P = "/root/autodl-tmp/src/logs/stcmds_standard3139_acoustic_listwise_predictions_20260729.jsonl.gz"
with gzip.open(P, "rt", encoding="utf-8") as f:
    r = json.loads(f.readline())
print("candidates type:", type(r.get("candidates")).__name__, "len:", len(r.get("candidates") or []))
c = (r.get("candidates") or [None])[0]
print("candidate[0] keys:", sorted(c.keys()) if isinstance(c, dict) else c)
print("candidate_evidence sample:", json.dumps((r.get("input") or {}).get("candidate_evidence"), ensure_ascii=False)[:400])
print("uncertain_spans sample:", json.dumps(r.get("uncertain_spans"), ensure_ascii=False)[:300])
print("id:", r.get("id"), "| asr_top1:", (r.get("asr_top1") or "")[:60])
print("prediction:", (r.get("prediction") or "")[:60])
print("reference :", (r.get("reference") or "")[:60])
print("hotwords  :", json.dumps((r.get("input") or {}).get("hotwords"), ensure_ascii=False))
print()
print("=== ST-CMDS hotword annotation dirs ===")
for root in glob.glob("/root/autodl-tmp/datasets/stcmds*") + glob.glob("/root/autodl-tmp/datasets/stcmds/*"):
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "hotword" or "hotword" in dirnames:
            for p in glob.glob(dirpath + "/*"):
                print(" ", p)
            for d in list(dirnames):
                if d == "hotword":
                    for dp, dn, fn in os.walk(os.path.join(dirpath, d)):
                        if dp.count(os.sep) - dirpath.count(os.sep) <= 2:
                            print("   ", dp, sorted(fn)[:8])
        dirnames[:] = [d for d in dirnames if d not in ("wav", "hs", "keywords-audios", "keywords-hs", "audio")]
