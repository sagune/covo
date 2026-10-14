#!/usr/bin/env python3
"""Third sweep: can the forced-CTC override be made safe everywhere by (a) blending
it with the front-end's own length-aware score `total_score` (rank-1 == 1.0), or
(b) restricting it to the decoder's own trust region (total_score >= rho * best)?

Rationale: `exact_weighted_score` is an inert 0.0 field on every dataset, so the
shipped reranker degenerates to `argmax(asr_score)` -- a *total* CTC log-likelihood
that is biased towards short candidates.  On ST-CMDS that shows up as -0.81 chars on
harmed rows; on AISHELL/THCHS-30 the pool length spread is small enough to hide it.
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


def zscores(vals):
    if len(vals) < 2:
        return [0.0] * len(vals)
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    s = var ** 0.5
    return [0.0] * len(vals) if s <= 1e-12 else [(v - m) / s for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ctc-scores", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
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
            s = num(c, "asr_score", -1e9)
            d[norm(c["text"])] = s / n if n else s
        ctc[str(r["id"])] = d

    rows = [json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = []
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
            pool.append(dict(text=t, total=num(c, "total_score"), ctc=num(c, "asr_score"), ctc_ln=scores[t]))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        zc = zscores([c["ctc_ln"] for c in sub])
        zt = zscores([c["total"] for c in sub])
        for i, c in enumerate(sub):
            c["zc"], c["zt"] = zc[i], zt[i]
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        cache.append(dict(uid=uid, old=old, ref=ref, sub=sub, hw=hw, keys=desig.get(uid, [])))

    chars = sum(len(c["ref"]) for c in cache)
    men = sum(len(c["keys"]) for c in cache)
    print("# %s   rows=%d chars=%d mentions=%d" % (a.label, len(cache), chars, men))

    def alpha_blend(al):
        def f(old, sub, hw):
            return max(sub, key=lambda c: al * c["zc"] + (1 - al) * c["zt"])["text"]
        return f

    def trust(rho):
        def f(old, sub, hw):
            top = max(c["total"] for c in sub)
            ok = [c for c in sub if c["total"] >= rho * top]
            return max(ok, key=lambda c: c["zc"])["text"]
        return f

    def blended_trust(rho, al):
        def f(old, sub, hw):
            top = max(c["total"] for c in sub)
            ok = [c for c in sub if c["total"] >= rho * top]
            return max(ok, key=lambda c: al * c["zc"] + (1 - al) * c["zt"])["text"]
        return f

    policies = [("identity", lambda o, s, h: o)]
    for al in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        policies.append(("blend a=%.1f (ctc_ln,total)" % al, alpha_blend(al)))
    for rho in (0.95, 0.9, 0.8, 0.6, 0.4):
        policies.append(("trust rho=%.2f -> ctc_ln" % rho, trust(rho)))
    for rho in (0.9, 0.7):
        policies.append(("trust rho=%.2f + blend a=0.5" % rho, blended_trust(rho, 0.5)))

    print("%-30s %8s %9s %8s %8s %6s %6s" % ("policy", "CER%", "dCER pp", "recall%", "drec pp", "spur", "chg%"))
    base = None
    for name, fn in policies:
        e = rec = spur = changed = 0
        for c in cache:
            new = fn(c["old"], c["sub"], c["hw"])
            e += edit_distance(list(c["ref"]), list(new))
            rec += sum(1 for k in c["keys"] if k in new)
            spur += sum(1 for h in c["hw"] if h in new and h not in c["ref"])
            changed += new != c["old"]
        cer, rcl = 100 * e / chars, 100 * rec / men
        if base is None:
            base = (cer, rcl)
        print("%-30s %8.4f %+9.4f %8.2f %+8.2f %6d %6.1f" % (
            name, cer, cer - base[0], rcl, rcl - base[1], spur, 100 * changed / len(cache)))


if __name__ == "__main__":
    main()
