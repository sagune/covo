#!/usr/bin/env python3
"""Diagnostic: does the protection set protect spurious hotwords?

`protect` is built from the front-end top-1 itself: every prompt hotword that
appears in `old`.  On a domain with a high spurious-insertion rate that means the
hard constraint can lock in an error, which would explain why the constrained
variants of the learned ranker give up almost all of their ST-CMDS gain.

This counts, for every utterance whose reference is known:
  * how many prompt hotwords the top-1 contains, and how many of them are spurious
    (present in the top-1 but absent from the reference)
  * the same for a given reranked file
so the two can be compared directly.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm


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
    ap.add_argument("--evidence", required=True, help="pre-rerank evidence (defines top-1 and protection)")
    ap.add_argument("--reranked", action="append", default=[], help="LABEL=PATH post-rerank evidence")
    a = ap.parse_args()

    pre = {}
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            pre[str(r.get("id"))] = r

    def stats(rows_iter, tag):
        n = tot_hw = spurious = locked_spurious = dropped = dropped_spurious = 0
        for r in rows_iter:
            src = pre.get(str(r.get("id")))
            if src is None:
                continue
            ref = norm(r.get("reference") or src.get("reference") or "")
            if not ref:
                continue
            inp = r.get("input") or {}
            top = norm(inp.get("asr_top1") or "")
            sinp = src.get("input") or {}
            old = norm(sinp.get("asr_top1") or "")
            prot_old = [h for h in texts_of(sinp.get("prompt_hotwords")) if h in old]
            n += 1
            tot_hw += len(prot_old)
            spurious += sum(1 for h in prot_old if h not in ref)
            locked_spurious += sum(1 for h in prot_old if h not in ref and h in top)
            lost = [h for h in prot_old if h not in top]
            dropped += len(lost)
            dropped_spurious += sum(1 for h in lost if h not in ref)
        print("%-26s rows=%5d | protected hotwords %6d | of which spurious %5d (%.2f%%) | still present %6d | dropped %5d | dropped-and-spurious %5d"
              % (tag, n, tot_hw, spurious, 100 * spurious / max(1, tot_hw), tot_hw - dropped, dropped, dropped_spurious))

    print("### protection built from the front-end top-1, and what each rerank does to it")
    stats((r for r in (json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip())),
          "pre-rerank (identity)")
    for item in a.reranked:
        label, _, path = item.partition("=")
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print("%-26s (not present)" % label)
            continue
        stats((json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()), label)


if __name__ == "__main__":
    main()
