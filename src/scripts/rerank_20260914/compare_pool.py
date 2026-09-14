#!/usr/bin/env python3
"""Compare pool ceilings: original dev evidence vs base rerun vs relaxed rerun."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
ORIG = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def analyse(path, label):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    TC = MEN = 0
    e_top = e_orc = 0
    h_top = h_orc = 0
    pool_sizes = []
    no_pool = 0
    nokws = 0
    for r in rows:
        uid = pos2utt[int(r["id"])] if str(r.get("id", "")).isdigit() and int(r["id"]) < len(pos2utt) else str(r.get("id"))
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        pool = texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest"))
        if not ref or not pool:
            continue
        TC += len(ref)
        e_top += edit_distance(list(ref), list(top))
        e_orc += min(edit_distance(list(ref), list(c)) for c in pool)
        pool_sizes.append(len(pool))
        hw = set(texts_of(inp.get("hotwords")))
        for k in desig.get(uid, []):
            MEN += 1
            h_top += k in top
            inpool = any(k in c for c in pool)
            h_orc += inpool
            if not inpool:
                no_pool += 1
                if not any(k == h or k in h for h in hw):
                    nokws += 1
    print("%-26s rows=%4d | top-1 CER %7.4f%% recall %6.2f%% | POOL ORACLE CER %7.4f%% recall %6.2f%% | mean pool %.2f | pool-miss %d (unretrieved %d)" % (
        label, len(rows), 100 * e_top / TC, 100 * h_top / MEN, 100 * e_orc / TC, 100 * h_orc / MEN,
        sum(pool_sizes) / len(pool_sizes), no_pool, nokws))
    return dict(cer_top=e_top / TC, rec_top=h_top / MEN, cer_oracle=e_orc / TC, rec_oracle=h_orc / MEN)


print("=== pool ceiling comparison (AISHELL dev, 599 designated mentions) ===")
analyse(ORIG, "original (2026-09-08)")
for tag in ("base", "relax"):
    p = R / ("dev_%s.jsonl" % tag)
    if p.exists() and p.stat().st_size > 0:
        analyse(p, "rerun %s" % tag)
    else:
        print("%-26s (not present)" % ("rerun %s" % tag))
