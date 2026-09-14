#!/usr/bin/env python3
"""Interface audit: does the reranked top-1 stay consistent with what COVO is shown?

`bridge prepare` renders three things from one record:
  * asr_top1                      -> the reranked text (apply_rerank_v2 rewrites it)
  * the n-best block              -> `input.nbest` in the reranked order, and for each
                                     line it prints the ORIGINAL `rank=`, `score=` and
                                     `asr=` looked up from cbwhisper.candidates
  * the "candidate scores" block  -> cbwhisper.candidates[0:max_candidates_with_scores]
                                     in ORIGINAL rank order

So if the reranked winner originally sat below the candidate-scores cutoff, COVO sees
a first n-best line whose own metadata says rank=N with a low score, while the
candidate block still leads with a different text.  This counts how often that
mismatch occurs, from the products only (no model, no bridge run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

CUT = 8  # --max-candidates-with-scores in every arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--pre", action="append", default=[], help="LABEL=ORIGINAL_EVIDENCE for the input side")
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    pre = {}
    for item in a.pre:
        label, _, path = item.partition("=")
        m = {}
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                i = int(r["id"])
                if i < len(pos2utt):
                    m[pos2utt[i]] = r
        pre[label] = m

    print("%-34s %7s %11s %11s %11s %11s" % (
        "run", "rows", "winner<cut", "winner>cut", "rank shift", "first-line==own_rank1"))
    for item in a.records:
        label, _, path = item.partition("=")
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print("%-34s (not present)" % label)
            continue
        n = in_cut = out_cut = 0
        shifts = []
        first_ok = 0
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            uid = r.get("id")
            inp = r.get("input") or {}
            top = norm(inp.get("asr_top1") or "")
            cands = ((inp.get("cbwhisper") or {}).get("candidates")) or []
            if not top or not cands:
                continue
            ranks = {}
            for c in cands:
                if c.get("text"):
                    ranks.setdefault(norm(c["text"]), int(c.get("rank", 999)))
            n += 1
            wr = ranks.get(top)
            if wr is None:
                continue
            if wr <= CUT:
                in_cut += 1
            else:
                out_cut += 1
            # original rank of the record's own rank-1 candidate
            r1 = min(ranks.values()) if ranks else 1
            first_ok += int(wr == r1)
            if wr != r1:
                shifts.append(wr - r1)
        avg = sum(shifts) / len(shifts) if shifts else 0.0
        print("%-34s %7d %11d %11d %11.2f %11.1f%%" % (
            label, n, in_cut, out_cut, avg, 100 * first_ok / n if n else 0))


if __name__ == "__main__":
    main()
