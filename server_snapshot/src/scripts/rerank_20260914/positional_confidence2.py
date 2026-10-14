#!/usr/bin/env python3
"""Does PER-CHARACTER ACOUSTIC confidence beat the 50% break-even?

Results 15 established the framing: at the character level there are ~2700 wrong characters
among ~56k positions, correcting a wrong character gains 1 and changing a right one loses 1,
so the decision breaks even at 50% precision.  The best available signal (N-best agreement)
reached 46.7% -- close, short.  The hypothesis was that the front end's per-candidate
`asr_score` is a MEAN and therefore destroys the shape we need.

`score_sensevoice_tokstats.py` now exports the per-character acoustic log-probability of the
top-1 (Viterbi forced alignment on the CTC lattice, aligned on normalised text).  This tests
whether that signal clears the bar.

    positional_confidence2.py --tokstats <file>
"""
import argparse
import difflib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DEF = R / "stcmds_va_tokstats.jsonl"


def opcodes(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def alt_chars(top, cands, p):
    """for each position of `top`, the best-weighted alternative character from `cands`"""
    alts = [None] * len(top)
    w = [0.0] * len(top)
    for ci, c in enumerate(cands):
        txt = norm(c.get("norm_text") or c.get("text") or "")
        for tag, i1, i2, j1, j2 in opcodes(top, txt):
            if tag == "equal":
                continue
            for i in range(i1, min(i2, len(top))):
                ch = txt[j1] if j1 < len(txt) else None
                if ch and ch != top[i] and p[ci] > w[i]:
                    w[i], alts[i] = p[ci], ch
    return alts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokstats", default=str(DEF))
    a = ap.parse_args()

    rows = []
    for line in Path(a.tokstats).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        cands = ((inp.get("cbwhisper") or {}).get("candidates") or [])
        if not cands:
            continue
        top = norm(cands[0].get("norm_text") or cands[0].get("text") or "")
        ref = norm(r.get("reference") or "")
        if not top or not ref:
            continue
        logps = cands[0].get("tok_logps")
        if not logps or len(logps) != len(top):
            continue                      # need per-character scores aligned 1:1
        sc = [c.get("asr_score") or 0.0 for c in cands]
        mx = max(sc)
        ex = [math.exp(s - mx) for s in sc]
        z = sum(ex)
        p = [e / z for e in ex]
        alts = alt_chars(top, cands, p)
        wrong = [True] * len(top)
        for tag, i1, i2, j1, j2 in opcodes(top, ref):
            if tag == "equal":
                for k in range(i2 - i1):
                    wrong[i1 + k] = False
        rows.append(dict(top=top, ref=ref, logp=logps, wrong=wrong, alt=alts,
                         e0=edit_distance(list(ref), list(top)), n=len(top)))

    TOTAL_CH = sum(len(r["ref"]) for r in rows)
    BASE_ERR = sum(r["e0"] for r in rows)
    pos = [(r["logp"][i], r["wrong"][i]) for r in rows for i in range(r["n"])]
    n_bad = sum(1 for _, w in pos if w)
    print("rows %d | characters %d | WRONG %d (%.2f%%)" % (len(rows), len(pos), n_bad, 100.0 * n_bad / len(pos)))
    print("front-end CER %.4f%%   target 4.5000%%" % (100.0 * BASE_ERR / TOTAL_CH))

    # AUC of per-character acoustic log-prob for "this character is wrong"
    pairs = sorted(pos)
    labels = [w for _, w in pos]
    rank, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        ar = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            if pairs[k][1]:
                rank += ar
        i = j + 1
    np_, nn_ = n_bad, len(pos) - n_bad
    auc_low = (rank - np_ * (np_ + 1) / 2.0) / (np_ * nn_)      # low logp => wrong
    print("AUC (per-character acoustic log-prob -> 'this char is wrong'): %.3f"
          % auc_low)
    print("  previous best signal (N-best agreement, per character): 0.851")
    print()
    print("  %-9s %-8s %-10s %-8s %s" % ("thresh", "changed", "precision", "recall", "corpus CER"))
    best = (None, 1e9)
    for thr in (-6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.1, 0.01):
        err = 0
        ch = corr = 0
        for r in rows:
            out = list(r["top"])
            for i2 in range(r["n"]):
                if r["logp"][i2] < thr and r["alt"][i2]:
                    out[i2] = r["alt"][i2]
                    ch += 1
                    if r["wrong"][i2]:
                        corr += 1
            err += edit_distance(list(r["ref"]), out)
        cer = 100.0 * err / TOTAL_CH
        prec = 100.0 * corr / max(1, ch)
        rec = 100.0 * corr / max(1, n_bad)
        note = " <= TARGET" if cer <= 4.5 else ""
        print("  %-9.2f %-8d %-9.1f%% %-7.1f%% %8.4f%%%s" % (thr, ch, prec, rec, cer, note))
        if cer < best[1]:
            best = (thr, cer)
    print()
    print("best threshold %.2f -> CER %.4f%%   (front end %.4f%%, break-even precision 50%%)"
          % (best[0], best[1], 100.0 * BASE_ERR / TOTAL_CH))


if __name__ == "__main__":
    main()
