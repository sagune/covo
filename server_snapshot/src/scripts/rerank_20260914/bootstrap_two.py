#!/usr/bin/env python3
"""Paired bootstrap between two evidence files (identity vs any reranked policy).

Unlike bootstrap_delta.py this does not re-derive policies: it compares the top-1 of
`--pre` with the top-1 of `--post` row by row, so it works for any reranked file
including the learned ranker's output.  Reports the 95% interval of dCER and
drecall and the fraction of resamples in which the post policy is no worse on both.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="reference policy (e.g. the front-end top-1)")
    ap.add_argument("--post", required=True, help="policy under test")
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = defaultdict(list)
    for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig[f[1].strip()].append(norm(f[0]))

    def load(path, by_pos):
        out = {}
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = str(r.get("id"))
            if by_pos and key.isdigit() and int(key) < len(pos2utt):
                key = pos2utt[int(key)]
            out[key] = r
        return out

    pre = load(a.pre, True)
    post = load(a.post, True)
    per = []
    for k, r in post.items():
        s = pre.get(k)
        if s is None:
            continue
        ref = norm(r.get("reference") or s.get("reference") or "")
        if not ref:
            continue
        t_pre = norm(((s.get("input") or {}).get("asr_top1")) or "")
        t_post = norm(((r.get("input") or {}).get("asr_top1")) or "")
        keys = desig.get(k, [])
        per.append(dict(chars=len(ref),
                        e0=edit_distance(list(ref), list(t_pre)),
                        e1=edit_distance(list(ref), list(t_post)),
                        m=len(keys),
                        r0=sum(1 for x in keys if x in t_pre),
                        r1=sum(1 for x in keys if x in t_post)))
    if not per:
        print("no comparable rows"); return

    def agg(sample):
        c = sum(x["chars"] for x in sample)
        m = sum(x["m"] for x in sample)
        return (100 * sum(x["e0"] for x in sample) / c, 100 * sum(x["e1"] for x in sample) / c,
                100 * sum(x["r0"] for x in sample) / m if m else 0.0,
                100 * sum(x["r1"] for x in sample) / m if m else 0.0)

    c0, c1, r0, r1 = agg(per)
    n = len(per)
    rng = random.Random(a.seed)
    dc, dr, both = [], [], 0
    for _ in range(a.draws):
        s = [per[rng.randrange(n)] for _ in range(n)]
        x0, x1, y0, y1 = agg(s)
        dc.append(x1 - x0)
        dr.append(y1 - y0)
        if (x1 - x0) <= 0 and (y1 - y0) >= 0:
            both += 1
    dc.sort(); dr.sort()
    lo, hi = dc[int(0.025 * a.draws)], dc[int(0.975 * a.draws)]
    rlo, rhi = dr[int(0.025 * a.draws)], dr[int(0.975 * a.draws)]
    print("# %s   rows=%d chars=%d mentions=%d draws=%d" % (a.label, n,
          sum(x["chars"] for x in per), sum(x["m"] for x in per), a.draws))
    print("  pre  (top-1)      CER %.4f%%  recall %.2f%%" % (c0, r0))
    print("  post (reranked)   CER %.4f%%  recall %.2f%%" % (c1, r1))
    print("  dCER %+.4f pp [%+.4f, %+.4f]   drecall %+.2f pp [%+.2f, %+.2f]   P(both non-worse) %.3f"
          % (c1 - c0, lo, hi, r1 - r0, rlo, rhi, both / a.draws))


if __name__ == "__main__":
    main()
