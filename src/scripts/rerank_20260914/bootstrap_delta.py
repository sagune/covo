#!/usr/bin/env python3
"""Paired bootstrap over utterances for the reranker's CER / recall deltas.

The fixed reranker's in-domain gains are small (-0.005 to -0.12 pp), so it matters
whether they are distinguishable from zero and whether they survive resampling.  For
each pool this resamples utterances with replacement (B=2000) and reports the 95%
interval of
    dCER    = CER(fixed) - CER(reference policy)
    drecall = recall(fixed) - recall(reference policy)
plus the fraction of draws in which the fixed policy is no worse on *both* axes.

Reference policies:
  front-end top-1  -- the CB-SenseVoice hypothesis before any reranking
  shipped reranker -- max(exact_weighted_score, raw forced-CTC total) with no floor
"""
import argparse
import json
import random
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
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
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
            raw = num(c, "asr_score", -1e9)
            d[norm(c["text"])] = (raw, raw / n if n else raw)
        ctc[str(r["id"])] = d

    rows = [json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
    per = []
    for r in rows:
        idx = int(r["id"])
        uid = pos2utt[idx] if idx < len(pos2utt) else None
        inp = r.get("input") or {}
        old = norm(inp.get("asr_top1") or "")
        ref = norm(r.get("reference") or "")
        scores = ctc.get(uid) if uid else None
        pool = []
        for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
            if not isinstance(c, dict) or not c.get("text"):
                continue
            t = norm(c["text"])
            if not scores or t not in scores:
                continue
            raw, ln = scores[t]
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=raw, ctc_ln=ln))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        keep = [c for c in sub if len(c["text"]) >= len(old)] or sub

        ship = max(sub, key=lambda c: (c["exact"], c["ctc"]))["text"]
        fixd = max(keep, key=lambda c: (c["exact"], c["ctc_ln"]))["text"]
        keys = desig.get(uid, [])
        per.append(dict(
            chars=len(ref),
            e_old=edit_distance(list(ref), list(old)),
            e_ship=edit_distance(list(ref), list(ship)),
            e_fix=edit_distance(list(ref), list(fixd)),
            m=len(keys),
            r_old=sum(1 for k in keys if k in old),
            r_ship=sum(1 for k in keys if k in ship),
            r_fix=sum(1 for k in keys if k in fixd),
        ))

    def agg(sample, ekey, rkey):
        c = sum(x["chars"] for x in sample)
        m = sum(x["m"] for x in sample)
        return (100 * sum(x[ekey] for x in sample) / c,
                100 * sum(x[rkey] for x in sample) / m if m else 0.0)

    full_fix = agg(per, "e_fix", "r_fix")
    print("# %s   rows=%d chars=%d mentions=%d draws=%d" % (
        a.label, len(per), sum(x["chars"] for x in per), sum(x["m"] for x in per), a.draws))
    print("  FIXED reranker: CER %.4f%%  recall %.2f%%" % full_fix)

    rng = random.Random(a.seed)
    n = len(per)
    for name, ek, rk in (("front-end top-1", "e_old", "r_old"), ("shipped reranker", "e_ship", "r_ship")):
        bc, br = agg(per, ek, rk)
        dcer, drec, both = [], [], 0
        for _ in range(a.draws):
            s = [per[rng.randrange(n)] for _ in range(n)]
            fc, fr = agg(s, "e_fix", "r_fix")
            xc, xr = agg(s, ek, rk)
            dc, dr = fc - xc, fr - xr
            dcer.append(dc)
            drec.append(dr)
            if dc <= 0 and dr >= 0:
                both += 1
        dcer.sort()
        drec.sort()
        print("  vs %-18s base %.4f%%/%.2f%% | dCER %+7.4f pp [%+.4f, %+.4f] | drecall %+6.2f pp [%+.2f, %+.2f] | P(both non-worse) %.3f"
              % (name, bc, br, full_fix[0] - bc, dcer[int(0.025 * a.draws)], dcer[int(0.975 * a.draws)],
                 full_fix[1] - br, drec[int(0.025 * a.draws)], drec[int(0.975 * a.draws)], both / a.draws))


if __name__ == "__main__":
    main()
