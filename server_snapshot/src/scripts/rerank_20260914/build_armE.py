#!/usr/bin/env python3
"""Arm E input: reranked order (top-1 + nbest) combined with forced-CTC acoustic scores."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
ranked = {json.loads(l)["id"]: json.loads(l) for l in (R / "dev_reranked.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
scored = {json.loads(l)["id"]: json.loads(l) for l in (R / "dev_ctcscored.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

out = []
for k, r in ranked.items():
    s = scored[k]
    ctc_by_text = {}
    for c in (s.get("input", {}).get("cbwhisper") or {}).get("candidates") or []:
        if isinstance(c, dict) and c.get("text"):
            ctc_by_text[norm(c["text"])] = (c.get("asr_score"), c.get("asr_score_decoder"))
    new = json.loads(json.dumps(r, ensure_ascii=False))
    for c in (new.get("input", {}).get("cbwhisper") or {}).get("candidates") or []:
        t = norm(c.get("text") or "")
        if t in ctc_by_text:
            c["asr_score"], c["asr_score_decoder"] = ctc_by_text[t]
    out.append(new)

(R / "dev_reranked_ctcscored.jsonl").write_text(
    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
print("rows %d -> %s" % (len(out), R / "dev_reranked_ctcscored.jsonl"))
