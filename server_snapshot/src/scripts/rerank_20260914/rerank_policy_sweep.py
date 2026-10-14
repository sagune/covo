#!/usr/bin/env python3
"""Offline policy sweep for the front-end reranker (CPU only).

Re-scores a whole family of candidate-selection policies on one dataset's logged
evidence pool + forced-CTC scores.  The point is to find a single selection rule
that raises designated-hotword recall *and* lowers CER on every domain, instead of
the currently shipped lexicographic (exact_weighted_score, ctc) override which
helps AISHELL / THCHS-30 but costs 0.75pp CER on ST-CMDS.
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
        ctc[str(r["id"])] = {norm(c["text"]): num(c, "asr_score", -1e9) for c in cs if c.get("text")}

    rows = [json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---- policy definitions: (old, sub, hw, dist) -> new text -----------------
    def shipped(old, sub, hw):
        return max(sub, key=lambda c: (c["exact"], c["ctc"]))["text"]

    def ctc_exact(old, sub, hw):
        return max(sub, key=lambda c: (c["ctc"], c["exact"]))["text"]

    def ctc_only(old, sub, hw):
        return max(sub, key=lambda c: c["ctc"])["text"]

    def exact_only(old, sub, hw):
        return max(sub, key=lambda c: c["exact"])["text"]

    def _count(t, hw):
        return sum(1 for h in hw if h in t)

    def noloss(old, sub, hw):
        new = shipped(old, sub, hw)
        return new if _count(new, hw) >= _count(old, hw) else old

    def gain_only(old, sub, hw):
        new = shipped(old, sub, hw)
        return new if _count(new, hw) > _count(old, hw) else old

    def bounded(d):
        def f(old, sub, hw):
            ok = [c for c in sub if edit_distance(list(old), list(c["text"])) <= d]
            return max(ok, key=lambda c: (c["exact"], c["ctc"]))["text"] if ok else old
        return f

    def bounded_noloss(d):
        def f(old, sub, hw):
            ok = [c for c in sub if edit_distance(list(old), list(c["text"])) <= d]
            if not ok:
                return old
            new = max(ok, key=lambda c: (c["exact"], c["ctc"]))["text"]
            return new if _count(new, hw) >= _count(old, hw) else old
        return f

    policies = [
        ("identity (old top-1)", lambda o, s, h: o),
        ("shipped  max(exact,ctc)", shipped),
        ("         max(ctc,exact)", ctc_exact),
        ("         max(ctc)", ctc_only),
        ("         max(exact)", exact_only),
        ("gate: hotword-noloss", noloss),
        ("gate: hotword-gain-only", gain_only),
        ("bounded edit<=1", bounded(1)),
        ("bounded edit<=2", bounded(2)),
        ("bounded edit<=3", bounded(3)),
        ("bounded<=2 + noloss", bounded_noloss(2)),
        ("bounded<=3 + noloss", bounded_noloss(3)),
    ]

    # ---- per-row cache -------------------------------------------------------
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
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=scores[t]))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        cache.append(dict(
            uid=uid, old=old, ref=ref, sub=sub, pool=pool,
            hw=sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords")))),
            keys=desig.get(uid, []),
        ))

    chars = sum(len(c["ref"]) for c in cache)
    men = sum(len(c["keys"]) for c in cache)

    print("# %s   rows=%d  chars=%d  mentions=%d" % (a.label, len(cache), chars, men))
    print("%-26s %9s %9s %8s %8s %7s %7s" % ("policy", "CER%", "dCER pp", "recall%", "d rec pp", "spur", "chg%"))
    base = None
    for name, fn in policies:
        e = rec = spur = changed = 0
        for c in cache:
            new = fn(c["old"], c["sub"], c["hw"])
            e += edit_distance(list(c["ref"]), list(new))
            rec += sum(1 for k in c["keys"] if k in new)
            spur += sum(1 for h in c["hw"] if h in new and h not in c["ref"])
            changed += new != c["old"]
        cer = 100 * e / chars
        rcl = 100 * rec / men
        if base is None:
            base = (cer, rcl)
        print("%-26s %9.4f %+9.4f %8.2f %+8.2f %7d %7.1f" % (
            name, cer, cer - base[0], rcl, rcl - base[1], spur, 100 * changed / len(cache)))

    # ---- attribution of the shipped policy's changes --------------------------
    print("\n## shipped-policy change anatomy (vs identity)")
    buckets = {"gain": [0, 0], "neutral": [0, 0], "harm": [0, 0]}
    tie = 0
    rec_gain = rec_loss = 0
    spur_old_tot = spur_new_tot = 0
    for c in cache:
        old = c["old"]
        new = shipped(old, c["sub"], c["hw"])
        d0 = edit_distance(list(c["ref"]), list(old))
        d1 = edit_distance(list(c["ref"]), list(new))
        exacts = {round(x["exact"], 6) for x in c["sub"]}
        if len(exacts) < len(c["sub"]):
            tie += 1
        spur_old_tot += sum(1 for h in c["hw"] if h in old and h not in c["ref"])
        spur_new_tot += sum(1 for h in c["hw"] if h in new and h not in c["ref"])
        if new == old:
            continue
        k = "gain" if d1 < d0 else ("neutral" if d1 == d0 else "harm")
        buckets[k][0] += 1
        buckets[k][1] += d1 - d0
        o_rec = sum(1 for x in c["keys"] if x in old)
        n_rec = sum(1 for x in c["keys"] if x in new)
        rec_gain += max(0, n_rec - o_rec)
        rec_loss += max(0, o_rec - n_rec)
    for k, (n, dd) in buckets.items():
        print("  %-8s rows %5d | CER edits %+6d" % (k, n, dd))
    print("  ties on exact_weighted_score within sub: %d rows" % tie)
    print("  mention deltas on changed rows: +%d / -%d" % (rec_gain, rec_loss))
    print("  spurious hotword insertions: %d -> %d" % (spur_old_tot, spur_new_tot))


if __name__ == "__main__":
    main()
