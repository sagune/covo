#!/usr/bin/env python3
"""Build forced-CTC scorer inputs + audio manifests for ST-CMDS and THCHS-30."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

W = Path("/root/autodl-tmp/.dsh_checks/rerank")
W.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(name="stcmds",
         evidence=Path("/root/autodl-tmp/.dsh_checks/stcmds/stcmds_evidence.jsonl"),
         uttid=Path("/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid"),
         wavdir=Path("/root/autodl-tmp/datasets/stcmds/cb_sensevoice_heldout/wav/test")),
    dict(name="thchs30",
         evidence=Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"),
         uttid=Path("/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid"),
         wavdir=Path("/root/autodl-tmp/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/wav/test")),
]

for job in JOBS:
    name = job["name"]
    if not job["evidence"].exists():
        print("%s: MISSING evidence %s" % (name, job["evidence"]))
        continue
    if not job["wavdir"].is_dir():
        print("%s: MISSING wav dir %s" % (name, job["wavdir"]))
        continue
    uttids = [l.split()[0] for l in job["uttid"].read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    wavs = {p.stem: str(p) for p in job["wavdir"].rglob("*.wav")}
    rows = [json.loads(l) for l in job["evidence"].read_text(encoding="utf-8").splitlines() if l.strip()]
    inp_rows, man_rows, miss_wav, miss_cand = [], [], 0, 0
    for r in rows:
        idx = int(r["id"])
        uid = uttids[idx] if idx < len(uttids) else None
        if uid is None or uid not in wavs:
            miss_wav += 1
            continue
        cands = [str(c["text"]).strip() for c in ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
                 if isinstance(c, dict) and c.get("text")]
        if not cands:
            miss_cand += 1
            continue
        inp_rows.append(dict(id=uid, input=dict(nbest=cands[:16]), reference=r.get("reference"), split="test"))
        man_rows.append(dict(id=uid, wav=wavs[uid], reference=r.get("reference"), split="test"))
    (W / ("%s_ctc_input.jsonl" % name)).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in inp_rows), encoding="utf-8")
    (W / ("%s_manifest.jsonl" % name)).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in man_rows), encoding="utf-8")
    print("%s: evidence=%d usable=%d (no wav %d, no candidates %d) mean_cands=%.2f" % (
        name, len(rows), len(inp_rows), miss_wav, miss_cand,
        sum(len(x["input"]["nbest"]) for x in inp_rows) / max(len(inp_rows), 1)))
    print("   wav dir has %d files; uttid rows %d" % (len(wavs), len(uttids)))
