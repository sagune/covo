#!/usr/bin/env python3
"""Fixed front-end reranker (replaces apply_rerank_generic.py).

Two defects in the shipped reranker, both found by the policy sweeps:

1. `max(asr_score)` uses the *total* forced-CTC log-likelihood of each candidate.
   That score grows with the number of emitted tokens, so an unconstrained argmax
   silently prefers shorter candidates.  On ST-CMDS the overridden rows lose 0.81
   characters on average and the override costs +0.75pp CER.
2. Nothing stops the override from deleting content the front-end was already
   confident about, so the deletion bias is free to act.

Fix: order candidates by (exact_weighted_score, length-normalised forced-CTC) but
only among pool candidates that (a) still contain every hotword the front-end's own
top-1 already contained and (b) are not shorter than that top-1.  The acoustic
evidence may therefore re-rank or extend the hypothesis, never truncate it.  This is
reference-free (no transcript, no test dictionary, no confidence threshold).

Measured effect at the relaxed admission (front-end CER / designated recall):
    AISHELL dev  3.8101/90.15 -> 3.5257/93.66      (baseline -> fixed)
    THCHS-30     3.4681/86.52 -> 3.1982/91.21
    ST-CMDS      5.7921/92.43 -> 5.90xx/93.31 (admission cost only; rerank is neutral)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


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
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-floor", action="store_true", help="disable the no-shorten constraint")
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = {}
    for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig.setdefault(f[1].strip(), []).append(norm(f[0]))

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

    rows = [json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows, changed, skipped, floored = [], 0, 0, 0
    chars = men = eb = ea = hb = ha = 0
    for r in rows:
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
        if not pool:
            skipped += 1
            out_rows.append(r)
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        if not a.no_floor:
            keep = [c for c in sub if len(c["text"]) >= len(old)]
            if len(keep) != len(sub):
                floored += 1
            sub = keep or sub
        best = max(sub, key=lambda c: (c["exact"], c["ctc_ln"]))

        ref = norm(r.get("reference") or "")
        if ref and uid:
            chars += len(ref)
            eb += edit_distance(list(ref), list(old))
            ea += edit_distance(list(ref), list(best["text"]))
            for k in desig.get(uid, []):
                men += 1
                hb += k in old
                ha += k in best["text"]
        if best["text"] != old:
            changed += 1
        new = json.loads(json.dumps(r, ensure_ascii=False))
        new["input"]["asr_top1"] = best["text"]
        new["input"]["nbest"] = [best["text"]] + [t for t in (inp.get("nbest") or []) if norm(t) != best["text"]]
        out_rows.append(new)

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
    print("# %s" % a.label)
    print("  rows %d | top-1 changed %d (%.1f%%) | no-floor-triggered %d | skipped %d" % (
        len(out_rows), changed, 100 * changed / len(out_rows), floored, skipped))
    print("  front-end CER    : %.4f%% -> %.4f%%  (%+.4f pp)" % (
        100 * eb / chars, 100 * ea / chars, 100 * (eb - ea) / chars))
    if men:
        print("  designated recall: %.2f%% (%d/%d) -> %.2f%% (%d/%d)" % (
            100 * hb / men, hb, men, 100 * ha / men, ha, men))


if __name__ == "__main__":
    main()
