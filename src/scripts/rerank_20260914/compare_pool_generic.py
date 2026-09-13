#!/usr/bin/env python3
"""Generic pool-ceiling comparison: original vs rerun evidence for any dataset."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def analyse(path, label, pos2utt, desig):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    TC = MEN = 0
    e_top = e_orc = h_top = h_orc = 0
    pools = []
    no_pool = nokws = 0
    for r in rows:
        rid = str(r.get("id", ""))
        uid = pos2utt[int(rid)] if rid.isdigit() and int(rid) < len(pos2utt) else rid
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        pool = texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest"))
        if not ref or not pool:
            continue
        TC += len(ref)
        e_top += edit_distance(list(ref), list(top))
        e_orc += min(edit_distance(list(ref), list(c)) for c in pool)
        pools.append(len(pool))
        hw = set(texts_of(inp.get("hotwords")))
        for k in desig.get(uid, []):
            MEN += 1
            h_top += k in top
            inp_pool = any(k in c for c in pool)
            h_orc += inp_pool
            if not inp_pool:
                no_pool += 1
                if not any(k == h or k in h for h in hw):
                    nokws += 1
    print("%-28s rows=%4d | top-1 CER %7.4f%% recall %6.2f%% | POOL ORACLE CER %7.4f%% recall %6.2f%% | mean pool %.2f | pool-miss %d (unretrieved %d)" % (
        label, len(rows), 100 * e_top / max(TC, 1), 100 * h_top / max(MEN, 1),
        100 * e_orc / max(TC, 1), 100 * h_orc / max(MEN, 1),
        sum(pools) / max(len(pools), 1), no_pool, nokws))
    return dict(rows=len(rows), cer_top=e_top / max(TC, 1), rec_top=h_top / max(MEN, 1),
                cer_oracle=e_orc / max(TC, 1), rec_oracle=h_orc / max(MEN, 1), pool_miss=no_pool)


ap = argparse.ArgumentParser()
ap.add_argument("--uttid", required=True)
ap.add_argument("--aligned", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--evidence", action="append", required=True, help="LABEL=PATH")
a = ap.parse_args()

pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

print("=== %s ===" % a.label)
for item in a.evidence:
    label, _, path = item.partition("=")
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        print("%-28s (not present)" % label)
        continue
    analyse(path, label, pos2utt, desig)
