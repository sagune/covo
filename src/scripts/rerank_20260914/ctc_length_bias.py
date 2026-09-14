#!/usr/bin/env python3
"""Quantify the forced-CTC length bias directly, per dataset.

`asr_score` is the *total* CTC log-likelihood of a fixed token sequence, so it should
be almost perfectly anti-correlated with sequence length; dividing by the token count
should remove that.  This prints the pooled Pearson/Spearman correlation, the implied
per-token slope, and the pool length spread that decides how much damage an
unconstrained `argmax(asr_score)` can do.
"""
import argparse
import json
from pathlib import Path


def num(c, k, d=0.0):
    try:
        return float(c.get(k, d) or d)
    except (TypeError, ValueError):
        return d


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc-scores", action="append", required=True, help="LABEL=PATH")
    a = ap.parse_args()

    print("%-22s %9s %9s %9s %9s %9s %9s %9s" % (
        "dataset", "pairs", "r(len,LL)", "rho", "dLL/token", "r(len,LL/n)", "pool dlen", "pool std"))
    for item in a.ctc_scores:
        label, _, path = item.partition("=")
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print("%-22s (not present)" % label)
            continue
        lens, lls, nlls, spreads, stds = [], [], [], [], []
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
            row = []
            for c in cs:
                tok = c.get("token_ids") or []
                n = len(tok)
                if n == 0:
                    continue
                s = num(c, "asr_score", -1e9)
                lens.append(n)
                lls.append(s)
                nlls.append(s / n)
                row.append(n)
            if len(row) >= 2:
                spreads.append(max(row) - min(row))
                m = sum(row) / len(row)
                stds.append((sum((x - m) ** 2 for x in row) / len(row)) ** 0.5)
        print("%-22s %9d %9.4f %9.4f %9.4f %9.4f %9.2f %9.3f" % (
            label, len(lens), pearson(lens, lls), spearman(lens, lls), slope(lens, lls),
            pearson(lens, nlls), sum(spreads) / max(1, len(spreads)), sum(stds) / max(1, len(stds))))
    print("\n  r(len,LL) strongly negative  -> total log-likelihood penalises long candidates")
    print("  r(len,LL/n) ~ 0              -> per-token normalisation removes the bias")
    print("  pool dlen / std              -> how much freedom the bias has inside one pool")


if __name__ == "__main__":
    main()
