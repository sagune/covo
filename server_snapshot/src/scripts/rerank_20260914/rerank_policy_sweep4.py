#!/usr/bin/env python3
"""Fourth sweep: a reference-free safety constraint for the acoustic override.

The shipped tiebreak harm anatomy says the failure mode on ST-CMDS is *deletion*:
harmed rows lose 0.81 characters on average, while on AISHELL / THCHS-30 harmed rows
are flat or slightly longer.  A total (unnormalised) CTC log-likelihood is exactly
the score that would do this, because it grows with the number of emitted tokens.

So: forbid the override from shortening the hypothesis.  This needs no references, no
dictionaries and no confidence gate -- it only compares the candidate with the
front-end's own top-1.
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
            d[norm(c["text"])] = (s, s / n if n else s)
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
            raw, ln = scores[t]
            pool.append(dict(text=t, total=num(c, "total_score"), ctc=raw, ctc_ln=ln))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        cache.append(dict(uid=uid, old=old, ref=ref, sub=sub, hw=hw, keys=desig.get(uid, [])))

    chars = sum(len(c["ref"]) for c in cache)
    men = sum(len(c["keys"]) for c in cache)
    print("# %s   rows=%d chars=%d mentions=%d" % (a.label, len(cache), chars, men))

    def floor(delta, key):
        def f(old, sub, hw):
            ok = [c for c in sub if len(c["text"]) >= len(old) + delta]
            return max(ok, key=key)["text"] if ok else old
        return f

    def floor_startswith(delta):
        """no-deletion: every character of the old hypothesis must survive in order."""
        def f(old, sub, hw):
            def keeps(c):
                it = iter(c["text"])
                return all(ch in it for ch in old)
            ok = [c for c in sub if len(c["text"]) >= len(old) + delta and keeps(c)]
            return max(ok, key=lambda c: c["ctc"])["text"] if ok else old
        return f

    policies = [
        ("identity", lambda o, s, h: o),
        ("shipped: raw ctc (no floor)", lambda o, s, h: max(s, key=lambda c: c["ctc"])["text"]),
        ("raw ctc, floor d=+1", floor(1, lambda c: c["ctc"])),
        ("raw ctc, floor d=0", floor(0, lambda c: c["ctc"])),
        ("raw ctc, floor d=-1", floor(-1, lambda c: c["ctc"])),
        ("ctc_ln, floor d=+1", floor(1, lambda c: c["ctc_ln"])),
        ("ctc_ln, floor d=0", floor(0, lambda c: c["ctc_ln"])),
        ("no-deletion (d=0, raw)", floor_startswith(0)),
        ("no-deletion (d=0) + noloss", lambda o, s, h: (lambda n: n if sum(1 for x in h if x in n) >= sum(1 for x in h if x in o) else o)(
            floor_startswith(0)(o, s, h))),
    ]

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
