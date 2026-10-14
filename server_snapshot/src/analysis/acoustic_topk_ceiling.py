#!/usr/bin/env python
"""Does the acoustic model CONTAIN the fix?

The whole "couple the front end to the LLM" idea rests on one empirical claim:
at the characters where CB-SenseVoice's top-1 is wrong, the correct character is
still present in the acoustic model's own distribution -- it was merely not
argmaxed.  If that claim is false (true homophones) then no projection layer,
no Q-Former and no acoustic soft prompt can help, because there is nothing to
inject.

This script measures the claim directly from forced-alignment token statistics:

  * align the front-end top-1 against the reference (normalised text),
  * at every aligned position where they differ, ask whether the reference
    character appears in the model's own top-k at the exactly corresponding
    emission frame (`tok_top5`, specials and the top-1 itself filtered out),
  * report hit@k and the CER ceiling of an oracle that may substitute a
    top-5 alternative *only when that alternative is the reference character*
    (i.e. a selector that is perfect but bounded by the acoustic side).

The last number is the point of the exercise: it is the best CER any purely
acoustic-coupled selector could reach, so it upper-bounds the whole plan.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def align_pairs(hyp: str, ref: str, include_equal: bool = False):
    """Yield (index_into_hyp, hyp_char, ref_char, kind).

    `kind` is 'sub' for equal-length replacements (addressable by a per-character
    decision), 'del'/'ins' for length-changing blocks (NOT addressable: the
    acoustic model would have had to emit a character it never emitted), and
    'eq' for matching characters (only when `include_equal`, which callers that
    must account for the COST of a wrong edit need).
    """
    sm = SequenceMatcher(None, hyp, ref, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            if include_equal:
                for d in range(i2 - i1):
                    yield i1 + d, hyp[i1 + d], ref[j1 + d], "eq"
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for d in range(i2 - i1):
                yield i1 + d, hyp[i1 + d], ref[j1 + d], "sub"
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for d in range(n):
                yield i1 + d, hyp[i1 + d], ref[j1 + d], "sub"
            for d in range(n, i2 - i1):
                yield i1 + d, hyp[i1 + d], "", "del"
            for d in range(n, j2 - j1):
                yield -1, "", ref[j1 + d], "ins"
        elif tag == "delete":
            for i in range(i1, i2):
                yield i, hyp[i], "", "del"
        elif tag == "insert":
            for j in range(j1, j2):
                yield -1, "", ref[j], "ins"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="*_tokstats.jsonl with tok_top5")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    print("rows: %d" % len(rows))

    K = a.top_k
    hit = [0] * (K + 1)           # hit[k] = errors whose ref char is in top-k
    err_total = 0
    err_sub = 0
    err_indel = 0
    considered = 0                # rows with usable token stats
    # confidence-stratified: terciles of the top-1 char's own log-prob
    strat = []                    # (logp, in_topk_hit_or_None)
    # oracle ceiling: apply ref char at hit positions only
    ref_len_tot = 0
    base_err_tot = 0
    oracle_err_tot = 0
    oracle1_err_tot = 0
    n_pos = 0
    rank_hist = Counter()

    for r in rows:
        ref = r.get("reference") or ""
        inp = r.get("input") or {}
        cands = ((inp.get("cbwhisper") or {}).get("candidates")) or []
        if not cands:
            continue
        c0 = cands[0]
        hyp = c0.get("norm_text") or ""
        top5 = c0.get("tok_top5")
        logps = c0.get("tok_logps") or []
        if not top5 or len(top5) != len(hyp):
            continue
        considered += 1
        ref_len_tot += len(ref)
        base_err_tot += edit_distance(hyp, ref)

        hyp_chars = list(hyp)
        hyp1_chars = list(hyp)
        n_pos += len(hyp)
        for idx, hc, rc, kind in align_pairs(hyp, ref):
            if kind != "sub" or not rc or rc == hc:
                if kind in ("del", "ins"):
                    err_indel += 1
                continue
            err_sub += 1
            err_total += 1
            if idx < 0 or idx >= len(top5):
                continue
            alts = top5[idx] or []
            # position of the reference character inside the model's own top-5
            rank = None
            for k, ch in enumerate(alts[:K], 1):
                if ch == rc:
                    rank = k
                    break
            if rank is not None:
                rank_hist[rank] += 1
                for k in range(rank, K + 1):
                    hit[k] += 1
                hyp_chars[idx] = rc          # oracle substitution
                if rank == 1:
                    hyp1_chars[idx] = rc
            lp = logps[idx] if idx < len(logps) else None
            if lp is not None:
                strat.append((float(lp), rank is not None))
        oracle_err_tot += edit_distance("".join(hyp_chars), ref)
        oracle1_err_tot += edit_distance("".join(hyp1_chars), ref)

    print()
    print("rows with usable token stats : %d / %d" % (considered, len(rows)))
    print("aligned substitution errors  : %d" % err_sub)
    print("aligned indel errors (NOT addressable per-character): %d" % err_indel)
    print()
    print("=== does the acoustic model CONTAIN the fix at error positions? ===")
    print("%-6s %10s %10s" % ("top-k", "hits", "% of errors"))
    for k in range(1, K + 1):
        pct = 100.0 * hit[k] / err_sub if err_sub else 0.0
        print("k=%-4d %10d %9.2f%%" % (k, hit[k], pct))
    print()
    print("=== confidence stratification (top-1 char log-prob tercile) ===")
    strat.sort(key=lambda t: t[0])
    if strat:
        n = len(strat)
        for name, seg in (("low  (worst conf)", strat[: n // 3]),
                          ("mid ", strat[n // 3: 2 * n // 3]),
                          ("high (most conf) ", strat[2 * n // 3:])):
            if not seg:
                continue
            cov = 100.0 * sum(1 for _, h in seg if h) / len(seg)
            print("  %-18s n=%-6d ref-in-top%d = %5.2f%%   logp[%.2f, %.2f]"
                  % (name, len(seg), K, cov, seg[0][0], seg[-1][0]))
    print()
    print("=== CER ceiling of a PERFECT acoustic-side selector ===")
    print("  total aligned positions                     : %d" % n_pos)
    print("  front-end top-1 CER                         : %.4f%%"
          % (100.0 * base_err_tot / ref_len_tot if ref_len_tot else 0))
    print("  oracle with ONLY the runner-up (rank 1)     : %.4f%%"
          % (100.0 * oracle1_err_tot / ref_len_tot if ref_len_tot else 0))
    print("  oracle @ top-%d (acoustic upper bound)       : %.4f%%"
          % (K, 100.0 * oracle_err_tot / ref_len_tot if ref_len_tot else 0))
    print("    -> acoustics alone can recover at most %.4f pp of CER"
          % (100.0 * (base_err_tot - oracle_err_tot) / ref_len_tot if ref_len_tot else 0))
    print("    -> of which the SINGLE runner-up accounts for %.4f pp"
          % (100.0 * (base_err_tot - oracle1_err_tot) / ref_len_tot if ref_len_tot else 0))
    print()
    print("=== why the decision rule, not the information, is the bottleneck ===")
    print("  rank of the reference char among the model's own alternatives")
    for r in range(1, K + 1):
        if rank_hist[r]:
            print("    alt#%d : %6d  (%.1f%% of the %d addressable errors)"
                  % (r, rank_hist[r], 100.0 * rank_hist[r] / err_sub, err_sub))
    print("  coverage of {top-1} u {alt#1}      : %.2f%% of errors"
          % (100.0 * hit[1] / err_sub if err_sub else 0))
    print("  coverage of {top-1} u {alt#1..#5}  : %.2f%% of errors"
          % (100.0 * hit[K] / err_sub if err_sub else 0))
    print()
    print("  NAIVE rule 'always swap in alt#1'  : %d fixed / %d broken  -> precision %.2f%%"
          % (hit[1], n_pos - err_sub, 100.0 * hit[1] / n_pos if n_pos else 0))
    print("  break-even precision for this project is ~54%% -> the swap must be GATED.")
    print("  The gate is a JOINT function of (acoustic alternatives at the frame, text context);")
    print("  neither module can compute it alone: CTC has the acoustics but no context,")
    print("  the LLM has the context but no acoustics.  That function is the coupling component.")

    if a.out:
        payload = {
            "rows": len(rows), "considered": considered,
            "err_sub": err_sub, "err_indel": err_indel, "n_pos": n_pos,
            "hit_at_k": {k: hit[k] for k in range(1, K + 1)},
            "hit_pct_at_k": {k: (100.0 * hit[k] / err_sub if err_sub else 0.0)
                             for k in range(1, K + 1)},
            "rank_hist": {r: rank_hist[r] for r in range(1, K + 1)},
            "frontend_cer_pct": 100.0 * base_err_tot / ref_len_tot if ref_len_tot else 0.0,
            "oracle_alt1_cer_pct": 100.0 * oracle1_err_tot / ref_len_tot if ref_len_tot else 0.0,
            "oracle_topk_cer_pct": 100.0 * oracle_err_tot / ref_len_tot if ref_len_tot else 0.0,
            "naive_swap1_precision_pct": 100.0 * hit[1] / n_pos if n_pos else 0.0,
        }
        Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
