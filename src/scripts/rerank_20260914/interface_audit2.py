#!/usr/bin/env python3
"""Corrected interface audit (replaces interface_audit.py).

What the prompt actually contains, with the flags every arm uses
(`compact_evidence` defaults to True, so `format_candidates` at
cbsensevoice_covo_bridge.py:1140 is skipped and there is NO "candidate scores"
block):

    ASR top-1: <the reranked winner>
    Protected hotwords that must be preserved exactly: ...
    N-best with reliability labels (...):
      1. <reranked winner> | trusted_scored rank=<ORIGINAL rank> score=<ORIGINAL total_score> ...
      2. <second n-best entry> | trusted_scored rank=<ORIGINAL rank> score=<ORIGINAL total_score> ...

So the observable risk is a contradiction *inside* the n-best block: line 1 is the
transmitted best candidate but its own printed metadata can rank and score it below
line 2. This measures that directly, from the stored pre/post products.

Usage: --pre <pre-rerank evidence> --post <post-rerank evidence> (ids must match).
Counts are keyed on the raw record id, not on uttid.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm


def load(path):
    out = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            out[str(r.get("id"))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    pre = load(a.pre)
    post = load(a.post)

    n = rank_moved = inv = 0
    rank1_sum = 0
    examples = []
    for rid, r in post.items():
        src = pre.get(rid)
        if src is None:
            continue
        meta = {}
        for c in ((src.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []:
            if c.get("text"):
                meta.setdefault(norm(c["text"]),
                                (int(c.get("rank", 999)), float(c.get("total_score", 0.0))))
        nb = [norm(t) for t in ((r.get("input") or {}).get("nbest") or [])][:8]
        if len(nb) < 2 or nb[0] not in meta:
            continue
        n += 1
        r1, s1 = meta[nb[0]]
        r2, s2 = meta.get(nb[1], (999, -1.0))
        rank1_sum += r1
        if r1 != 1:
            rank_moved += 1
        if s1 < s2 - 1e-9:
            inv += 1
            if len(examples) < 3:
                examples.append((rid, r1, s1, r2, s2))
    if not n:
        print("%-42s (no comparable rows)" % a.label)
        return
    print("%-42s rows=%5d | line1 original rank != 1: %5d (%5.1f%%) | VISIBLE score inversion: %5d (%5.1f%%) | mean line1 rank %.2f"
          % (a.label, n, rank_moved, 100 * rank_moved / n, inv, 100 * inv / n, rank1_sum / n))
    for rid, r1, s1, r2, s2 in examples:
        print("        id=%-14s line1 rank=%3d score=%+.3f   vs   line2 rank=%3d score=%+.3f" % (rid, r1, s1, r2, s2))


if __name__ == "__main__":
    main()
