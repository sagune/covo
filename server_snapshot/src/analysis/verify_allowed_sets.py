#!/usr/bin/env python
"""Verify the constrained-decoding sets reproduce RESULTS section 17's coverage.

section 17 measured that the reference character lies inside the acoustic model's
own top-5 for **87.50%** of the 2625 addressable substitution errors.  That figure
was computed directly from `tok_top5` indexed by the NORMALISED hypothesis.

`build_allowed_sets.py` re-indexes those alternatives onto the RAW hypothesis (so
that position p means the same character in the encoder-state channel and in the
constrained-decoding channel).  If that re-indexing were wrong, coverage would fall
below 87.5% and the constrained decoder would be constraining the wrong characters.

So this script recomputes coverage from the produced allowed sets.  Agreement with
87.5% is the proof that both channels index identically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp")
sys.path.insert(0, str(REPO / "src" / "analysis"))
from acoustic_topk_ceiling import align_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="*_allowed_sets.jsonl")
    ap.add_argument("--input", required=True, help="evidence jsonl carrying `reference`")
    a = ap.parse_args()

    ref_by_id = {}
    for l in Path(a.input).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ref_by_id[str(r.get("id"))] = r.get("reference") or ""

    err = 0
    hit = [0, 0]        # [in set at all, reference is the FIRST alternative]
    n_rows = 0
    for l in Path(a.sets).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        rid = str(r.get("id"))
        if rid not in ref_by_id:
            continue
        top1 = r.get("top1") or ""
        topk = r.get("topk") or []
        ref = ref_by_id[rid]
        n_rows += 1
        for p, hc, rc, kind in align_pairs(top1, ref):
            if kind != "sub" or not rc or rc == hc or p < 0 or p >= len(topk):
                continue
            err += 1
            allowed = [hc] + list(topk[p])
            if rc in allowed:
                hit[0] += 1
            if topk[p] and topk[p][0] == rc:
                hit[1] += 1

    print("rows compared                 : %d" % n_rows)
    print("addressable substitution errs : %d" % err)
    if err:
        print("reference char in allowed set : %d  (%.2f%%)" % (hit[0], 100.0 * hit[0] / err))
        print("  ... is the FIRST alternative: %d  (%.2f%%)  <- the break-even comparison"
              % (hit[1], 100.0 * hit[1] / err))
    print()
    print("RESULTS section 17 reference  : 87.50%% in top-5, 58.82%% as first alternative")
    print("project break-even            : ~54%%")
    pct_all = 100.0 * hit[0] / err if err else 0.0
    pct_first = 100.0 * hit[1] / err if err else 0.0
    ok = err > 0 and 84.0 <= pct_all <= 90.0 and 55.0 <= pct_first <= 62.0
    print("VERDICT: %s" % ("PASS -- both central figures reproduce, so the encoder-state "
                           "channel and the constrained-decoding channel index the same "
                           "characters"
                           if ok else "FAIL -- figures do not match section 17"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
