#!/usr/bin/env python3
"""How informative is `exact_weighted_score` inside the pool, per dataset?

`exact_weighted_score` is the hotword exact-coverage statistic (cb_sensevoice.py:1341/2756).
The shipped reranker uses it as the PRIMARY sort key, so if it is constant inside most
pools the reranker silently degenerates to `argmax(forced-CTC)`.  This measures that.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="append", required=True, help="LABEL=PATH")
    a = ap.parse_args()

    print("%-24s %7s %11s %11s %11s %11s" % (
        "pool", "rows", "all-zero", "any-nonzero", "distinct>=2", "top1-is-max"))
    for item in a.evidence:
        label, _, path = item.partition("=")
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print("%-24s (not present)" % label)
            continue
        rows = allZ = anyNZ = distinct = topmax = n = 0
        vals = Counter()
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
            e = [float(c.get("exact_weighted_score") or 0.0) for c in cs if c.get("text")]
            if len(e) < 2:
                continue
            n += 1
            if all(v == 0.0 for v in e):
                allZ += 1
            if any(v != 0.0 for v in e):
                anyNZ += 1
            if len(set(e)) >= 2:
                distinct += 1
            if e[0] == max(e):
                topmax += 1
            for v in e:
                vals[round(v, 3)] += 1
        print("%-24s %7d %10.1f%% %10.1f%% %10.1f%% %10.1f%%" % (
            label, n, 100 * allZ / n, 100 * anyNZ / n, 100 * distinct / n, 100 * topmax / n))
        print("      most common exact_weighted_score values: %s" % (
            ", ".join("%s x%d" % (k, v) for k, v in vals.most_common(5))))


if __name__ == "__main__":
    main()
