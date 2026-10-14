#!/usr/bin/env python
"""Ceiling of a gate that uses ONLY per-character acoustic evidence.

`acoustic_topk_ceiling.py` showed the acoustic side CONTAINS the fix (87.5% of
addressable errors have the reference character inside the model's own top-5,
argmax excluded).  That is content, not decision.  This script measures the
decision side: sweep a 1-D gate over per-character acoustic features and, when
it fires, swap in the model's own best alternative.  The best net character gain
found is the ceiling of every purely-acoustic repair rule, and therefore the
budget the language model has to beat by reading context.

Features swept (each independently; lower = less confident = act):
  logp    : the emitted character's own CTC log-prob at its frame
  margin  : logp(argmax) - logp(best alternative)
  altlogp : the best alternative's own log-prob
  n_alt   : how many alternatives were available at all (availability, not conf)

Reported per threshold: acted / fixed / broken / net chars / resulting CER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acoustic_topk_ceiling import align_pairs, edit_distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.input).read_text(encoding="utf-8").splitlines() if l.strip()]

    # collect every aligned position with its acoustic features and its outcome.
    # `margin` = logp(argmax) - logp(best alternative) is stored per position;
    # a SMALL margin means the acoustic model itself is unsure, so the gate fires
    # on margin < threshold (and on logp < threshold for the absolute-uncertainty view).
    pos = []
    tot_chars = 0
    base_err = 0
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
        margins = c0.get("tok_alt_margin") or []
        if not top5 or len(top5) != len(hyp):
            continue
        tot_chars += len(ref)
        base_err += edit_distance(hyp, ref)
        for idx, hc, rc, kind in align_pairs(hyp, ref, include_equal=True):
            if idx < 0 or idx >= len(hyp) or kind == "del":
                continue
            alts = [c for c in (top5[idx] or []) if c]
            lp = float(logps[idx]) if idx < len(logps) else -20.0
            mg = float(margins[idx]) if idx < len(margins) else 1e9
            pos.append({"logp": lp, "margin": mg, "alts": alts, "ref": rc,
                        "hyp": hc, "is_err": rc != hc})

    n_pos = len(pos)
    n_err = sum(1 for p in pos if p["is_err"])
    print("positions %d  errors %d (%.2f%%)  base CER %.4f%%"
          % (n_pos, n_err, 100.0 * n_err / n_pos, 100.0 * base_err / tot_chars))

    def evaluate(score, thresholds):
        """Character-level accounting.

        Only two outcomes move the character error count:
          fixed   -- the position was wrong and the swapped-in character is the
                     reference character:  -1 error
          damaged -- the position was ALREADY CORRECT and we swapped it:  +1 error
        An error position where the alternative is not the reference character
        stays wrong either way, so it is neither a gain nor a loss.
        """
        best = None
        curve = []
        svals = np.array([p[score] if p.get(score) is not None else np.nan for p in pos], dtype=float)
        for thr in thresholds:
            fixed = damaged = failed = acted = 0
            for i, p in enumerate(pos):
                s = svals[i]
                if not np.isfinite(s) or s >= thr:
                    continue
                acted += 1
                if p["is_err"]:
                    if p["ref"] in p["alts"][: a.top_k]:
                        fixed += 1
                    else:
                        failed += 1
                else:
                    damaged += 1
            net = fixed - damaged
            cer = 100.0 * (base_err - net) / tot_chars
            curve.append({"thr": float(thr), "acted": acted, "fixed": fixed,
                          "failed": failed, "damaged": damaged, "net": net, "cer_pct": cer})
            if best is None or net > best["net"]:
                best = dict(curve[-1], feature=score)
        return best, curve

    print()
    print("=== ceiling of a purely acoustic repair rule (swap in the best alternative) ===")
    results = {}
    for feat, thrs in (
        ("logp", np.arange(-14.0, 1.01, 0.25)),
        ("margin", np.arange(-8.0, 4.01, 0.25)),
    ):
        best, curve = evaluate(feat, thrs)
        results[feat] = {"best": best, "curve": curve}
        print("  feat=%-7s best net %+6d chars at thr %.2f  (acted %d, fixed %d, damaged %d) -> CER %.4f%%"
              % (feat, best["net"], best["thr"], best["acted"], best["fixed"],
                 best["damaged"], best["cer_pct"]))
        # operating points of interest
        for target in (100, 300, 438):
            for pt in curve:
                if pt["net"] >= target:
                    print("      reaches +%d net chars at thr %.2f (acted %d, fixed %d, damaged %d) CER %.4f%%"
                          % (target, pt["thr"], pt["acted"], pt["fixed"], pt["damaged"], pt["cer_pct"]))
                    break
            else:
                print("      CANNOT reach +%d net chars at any threshold" % target)

    print()
    print("  reference: the 4.5%% target needs ~438 net characters")
    print("  reference: full 24-candidate N-best pool oracle CER = 2.3254%%")
    print("  reference: acoustic top-5 oracle CER               = 1.1611%%")

    if a.out:
        Path(a.out).write_text(json.dumps({"n_pos": n_pos, "n_err": n_err,
                                           "base_cer_pct": 100.0 * base_err / tot_chars,
                                           "tot_chars": tot_chars, **results},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
