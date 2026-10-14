#!/usr/bin/env python3
"""Oracle-level cost of the no-shorten constraint.

The fix forbids the acoustic override from returning a candidate shorter than the
front-end top-1.  That is principled but not free: when the front-end over-generated,
only a shorter candidate can be right.  This measures the forfeited oracle headroom
(sum over rows of best_with_floor - best_without, in CER points) against the realised
benefit the constraint buys, so the trade-off can be stated honestly.
"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--uttid", required=True)
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    print("%-26s %7s %8s %9s %9s %9s %9s" % (
        "pool", "rows", "chars", "top1 CER", "oracle", "oracle|floor", "floor cost"))
    for item in a.evidence:
        label, _, path = item.partition("=")
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print("%-26s (not present)" % label)
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        chars = e_top = e_orc = e_flo = 0
        n_short_better = 0
        for r in rows:
            rid = str(r.get("id", ""))
            inp = r.get("input") or {}
            ref = norm(r.get("reference") or "")
            if not ref:
                continue
            uid = pos2utt[int(rid)] if rid.isdigit() else None
            if uid is None:
                continue
            old = norm(inp.get("asr_top1") or "")
            pool = texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest"))
            if not pool:
                continue
            chars += len(ref)
            d_old = edit_distance(list(ref), list(old))
            best = min(edit_distance(list(ref), list(c)) for c in pool)
            keep = [c for c in pool if len(c) >= len(old)] or [old]
            best_floor = min(edit_distance(list(ref), list(c)) for c in keep)
            e_top += d_old
            e_orc += best
            e_flo += best_floor
            if best < best_floor:
                n_short_better += 1
        print("%-26s %7d %8d %8.4f%% %8.4f%% %9.4f%% %8.4fpp  (%d rows need a shorter candidate)" % (
            label, len(rows), chars, 100 * e_top / chars, 100 * e_orc / chars,
            100 * e_flo / chars, 100 * (e_flo - e_orc) / chars, n_short_better))


if __name__ == "__main__":
    main()
