#!/usr/bin/env python3
"""Build a no-reorder variant: keep the learned top-1 but restore the beam's own
n-best order, so every line's printed rank/score matches its position.

This is the most conservative interface: the rescorer only replaces `asr_top1`, it
does not reorder the list COVO reads.  Section 2's arm C already showed that
reordering the n-best bought nothing over simply dropping the trust anchor
(2.5543 vs 2.5495 pp), so removing the reorder costs no known benefit while making
the prompt self-consistent.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="pre-rerank evidence (defines the beam order)")
    ap.add_argument("--reranked", required=True, help="post-rerank evidence (defines the new top-1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orig = {}
    for l in Path(a.original).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            orig[str(r.get("id"))] = r

    rows, changed = [], 0
    for l in Path(a.reranked).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        s = orig.get(str(r.get("id")))
        if s is not None:
            nb = (s.get("input") or {}).get("nbest")
            if nb:
                if (r.get("input") or {}).get("nbest") != nb:
                    changed += 1
                r["input"]["nbest"] = list(nb)
        rows.append(r)

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    print("wrote %d rows to %s (n-best order restored on %d)" % (len(rows), a.out, changed))


if __name__ == "__main__":
    main()
