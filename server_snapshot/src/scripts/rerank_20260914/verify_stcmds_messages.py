#!/usr/bin/env python3
import json
from pathlib import Path

P = Path("/root/autodl-tmp/.dsh_checks/stcmds/stcmds_selector.messages.jsonl")
rows = []
with P.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
print("rows:", len(rows))
r = rows[0]
inp = r.get("input") or {}
print("top keys :", sorted(r.keys()))
print("input keys:", sorted(inp.keys()))
print("messages :", len(r.get("messages") or []))
print("reference:", (r.get("reference") or "")[:60])
print("hotwords len=%d prompt_hotwords len=%d cands len=%d" % (
    len(inp.get("hotwords") or []), len(inp.get("prompt_hotwords") or []),
    len((inp.get("cbwhisper") or {}).get("candidates") or [])))
sys_msg = r["messages"][0]["content"]
print("system preview:", sys_msg[:180].replace("\n", " | "))
print()
counts = {"has_hotwords": 0, "has_prompt_hw": 0, "has_cands": 0, "has_reference": 0}
for x in rows:
    i = x.get("input") or {}
    counts["has_hotwords"] += bool(i.get("hotwords"))
    counts["has_prompt_hw"] += bool(i.get("prompt_hotwords"))
    counts["has_cands"] += bool((i.get("cbwhisper") or {}).get("candidates"))
    counts["has_reference"] += bool(x.get("reference"))
print("coverage over %d rows: %s" % (len(rows), counts))
print()
print("user prompt (first 1200 chars):")
print(r["messages"][1]["content"][:1200])
