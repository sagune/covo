#!/usr/bin/env python3
"""Second policy sweep: is the forced-CTC override's domain dependence a length bias?

`score_sensevoice_candidate_evidence.py` stores `asr_score` = *total* (unnormalised)
CTC log-likelihood, so `max(asr_score)` silently prefers the shortest candidate.
That is harmless where the pool's length spread is small and catastrophic where it
is not.  This sweep re-scores the same pools with length-normalised CTC scores and
with a hotword-set-frozen variant, and prints the length/edits anatomy of the
shipped policy's harmed rows.
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
            d[norm(c["text"])] = (s, s / n if n else s, n)
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
            raw, ln, ntok = scores[t]
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=raw, ctc_ln=ln, ntok=ntok))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        cache.append(dict(uid=uid, old=old, ref=ref, sub=sub, hw=hw, keys=desig.get(uid, []),
                          hwset=frozenset(h for h in hw if h in old)))

    chars = sum(len(c["ref"]) for c in cache)
    men = sum(len(c["keys"]) for c in cache)

    def _hwset(t, hw):
        return frozenset(h for h in hw if h in t)

    def pick(old, sub, hw, key, same_set=False):
        pool = sub
        if same_set:
            hs = _hwset(old, hw)
            pool = [c for c in sub if _hwset(c["text"], hw) == hs] or sub
        return max(pool, key=key)["text"]

    policies = [
        ("identity", lambda o, s, h: o),
        ("ctc raw (shipped tiebreak)", lambda o, s, h: pick(o, s, h, lambda c: (c["exact"], c["ctc"]))),
        ("ctc len-normalised", lambda o, s, h: pick(o, s, h, lambda c: (c["exact"], c["ctc_ln"]))),
        ("ctc len-norm only", lambda o, s, h: pick(o, s, h, lambda c: c["ctc_ln"])),
        ("ctc raw  + hwset frozen", lambda o, s, h: pick(o, s, h, lambda c: (c["exact"], c["ctc"]), True)),
        ("ctc len-norm + hwset frozen", lambda o, s, h: pick(o, s, h, lambda c: (c["exact"], c["ctc_ln"]), True)),
        ("ctc len-norm + noloss", lambda o, s, h: (lambda n: n if sum(1 for x in h if x in n) >= sum(1 for x in h if x in o) else o)(
            pick(o, s, h, lambda c: (c["exact"], c["ctc_ln"])))),
        ("ctc len-norm + hwset + noloss", lambda o, s, h: (lambda n: n if sum(1 for x in h if x in n) >= sum(1 for x in h if x in o) else o)(
            pick(o, s, h, lambda c: (c["exact"], c["ctc_ln"]), True))),
    ]

    print("# %s   rows=%d chars=%d mentions=%d" % (a.label, len(cache), chars, men))
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

    # ---- anatomy of the shipped policy: is the harm a length artefact? --------
    print("\n## anatomy of the raw-CTC override (vs identity)")
    from collections import Counter
    lens = Counter()
    rows_len = Counter()
    tot_dlen_gain = tot_dlen_harm = 0
    ng = nh = 0
    for c in cache:
        old = c["old"]
        new = pick(old, c["sub"], c["hw"], lambda x: (x["exact"], x["ctc"]))
        if new == old:
            continue
        d0 = edit_distance(list(c["ref"]), list(old))
        d1 = edit_distance(list(c["ref"]), list(new))
        dlen = len(new) - len(old)
        if d1 < d0:
            lens["gain"] += dlen
            rows_len["gain"] += 1
            tot_dlen_gain += dlen
            ng += 1
        elif d1 > d0:
            lens["harm"] += dlen
            rows_len["harm"] += 1
            tot_dlen_harm += dlen
            nh += 1
    print("  gain rows %d  mean len change %+.3f" % (rows_len["gain"], tot_dlen_gain / max(1, ng)))
    print("  harm rows %d  mean len change %+.3f" % (rows_len["harm"], tot_dlen_harm / max(1, nh)))
    print("  (a strongly negative harm mean == the override deletes characters)")


if __name__ == "__main__":
    main()
