#!/usr/bin/env python3
"""Build the scorer input: CB-SenseVoice 16-candidate pool presented as `nbest`."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
OUT = Path("/root/autodl-tmp/.dsh_checks/rerank")
OUT.mkdir(parents=True, exist_ok=True)

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]

rows = []
for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    inp = r.get("input") or {}
    cands = [str(c["text"]).strip() for c in (inp.get("cbwhisper") or {}).get("candidates") or []
             if isinstance(c, dict) and c.get("text")]
    if not cands:
        continue
    uid = pos2utt[int(r["id"])]
    rows.append(dict(id=uid, input=dict(nbest=cands[:16]), reference=r.get("reference"), split="dev"))

(OUT / "ctc_input.jsonl").write_text(
    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps(dict(rows=len(rows),
                      mean_candidates=sum(len(x["input"]["nbest"]) for x in rows) / len(rows)), ensure_ascii=False))
