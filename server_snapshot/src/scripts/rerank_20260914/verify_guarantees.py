#!/usr/bin/env python3
"""Verify the three formal guarantees of the fixed reranker.

Setup.  For one utterance let
    old  = the front-end's own hypothesis (asr_top1)
    P    = { prompt hotwords that occur in old }          (the protection set)
    sub  = { candidates that contain every phrase in P, and are not shorter than old }
    new  = argmax_{c in sub} ( exact_weighted_score(c), ctc_loglik(c)/len(c) )

Because `old` itself contains every phrase in P and len(old) >= len(old), we have
old in sub, hence:

  G1 (non-shortening)   len(new) >= len(old)
  G2 (coverage never drops)  exact_weighted_score(new) >= exact_weighted_score(old)
  G3 (protection respected)  every phrase in P still occurs in new

G2/G3 hold by construction (old is a feasible point of the argmax); G1 holds by the
feasibility restriction.  Note that none of them implies designated-hotword recall is
non-decreasing: `exact_weighted_score` measures the *injected prompt* hotwords, not
the held-out designated list.  This script checks all three on the real pools, and
counts how often the SHIPPED reranker (no feasibility restriction) breaks them.
"""
import argparse
import json
import sys
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


def num(c, k, d=0.0):
    try:
        return float(c.get(k, d) or d)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ctc-scores", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    ctc = {}
    for l in Path(a.ctc_scores).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
        d = {}
        for c in cs:
            if not c.get("text"):
                continue
            tok = c.get("token_ids") or []
            n = len(tok) if tok else max(1, len(norm(c["text"])))
            d[norm(c["text"])] = num(c, "asr_score", -1e9) / n if n else -1e9
        ctc[str(r["id"])] = d

    n = 0
    f = {"G1": 0, "G2": 0, "G3": 0}          # fixed reranker violations
    s = {"G1": 0, "G2": 0, "G3": 0}          # shipped reranker violations
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        idx = int(r["id"])
        uid = pos2utt[idx] if idx < len(pos2utt) else None
        inp = r.get("input") or {}
        old = norm(inp.get("asr_top1") or "")
        scores = ctc.get(uid) if uid else None
        pool = []
        for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
            if not isinstance(c, dict) or not c.get("text"):
                continue
            t = norm(c["text"])
            if not scores or t not in scores:
                continue
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc_ln=scores[t]))
        if not pool or not old or not uid:
            continue
        n += 1
        old_exact = 0.0
        for c in pool:
            if c["text"] == old:
                old_exact = c["exact"]
                break
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        keep = [c for c in sub if len(c["text"]) >= len(old)] or sub
        ship = max(sub, key=lambda c: (c["exact"], c["ctc_ln"]))["text"]
        fixd = max(keep, key=lambda c: (c["exact"], c["ctc_ln"]))["text"]

        for name, txt, bag in (("fix", fixd, f), ("ship", ship, s)):
            if len(txt) < len(old):
                bag["G1"] += 1
            new_exact = 0.0
            for c in pool:
                if c["text"] == txt:
                    new_exact = c["exact"]
                    break
            if new_exact < old_exact - 1e-9:
                bag["G2"] += 1
            if not all(p in txt for p in protect):
                bag["G3"] += 1

    print("# %-44s rows=%d" % (a.label, n))
    print("    FIXED  reranker violations:  G1(non-shortening) %d   G2(coverage) %d   G3(protection) %d"
          % (f["G1"], f["G2"], f["G3"]))
    print("    SHIPPED reranker violations: G1(non-shortening) %d   G2(coverage) %d   G3(protection) %d"
          % (s["G1"], s["G2"], s["G3"]))


if __name__ == "__main__":
    main()
