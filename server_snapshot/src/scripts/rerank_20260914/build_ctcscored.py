#!/usr/bin/env python3
"""Arm D input: replace the prompt-visible acoustic score with the unbiased forced-CTC one.

The bridge renders `asr={candidate.asr_score:}` into the N-best lines, and we measured
that this decoder score is anti-correlated with correctness inside the preserving
subset. This rewrites asr_score to the forced-CTC log-likelihood (matched by text),
keeping the original under `asr_score_decoder` for traceability. Everything else is
untouched, so the change is single-variable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

SRC = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl")
CTC = Path("/root/autodl-tmp/.dsh_checks/rerank/ctc_scores.jsonl")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
OUT = Path("/root/autodl-tmp/.dsh_checks/rerank/dev_ctcscored.jsonl")

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]

ctc = {}
for l in CTC.read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
    ctc[str(r["id"])] = {norm(c["text"]): float(c.get("asr_score") or 0.0) for c in cs if c.get("text")}

rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
out, replaced, missing = [], 0, 0
for r in rows:
    uid = pos2utt[int(r["id"])]
    scores = ctc.get(uid) or {}
    new = json.loads(json.dumps(r, ensure_ascii=False))
    for c in (new.get("input", {}).get("cbwhisper") or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        t = norm(c["text"])
        if t in scores:
            c["asr_score_decoder"] = c.get("asr_score")
            c["asr_score"] = scores[t]
            replaced += 1
        else:
            missing += 1
    out.append(new)

OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
print("rows %d | candidates rescored %d | unmatched %d" % (len(out), replaced, missing))
print("wrote", OUT)

s = out[0]["input"]["cbwhisper"]["candidates"][0]
print("example candidate keys:", sorted(s.keys()))
print("  asr_score (now forced-CTC):", round(s["asr_score"], 4), "| decoder:", round(s.get("asr_score_decoder", 0), 4))
