#!/usr/bin/env python3
"""Full key x floor grid for the front-end reranker.

`sweep4` showed the no-shorten floor fixes ST-CMDS for a pure forced-CTC key, while
`sweep5` showed it does nothing when `exact_weighted_score` stays the primary key --
and that dropping `exact` costs ~1.3pp designated recall on THCHS-30.  So the two
terms genuinely interact; this grid scores every ordering against every floor on
every dataset/admission pair so the final rule can be chosen from data rather than
from one dataset at a time.
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


KEYS = {
    "exact": lambda c: c["exact"],
    "ctc": lambda c: c["ctc"],
    "ctc_ln": lambda c: c["ctc_ln"],
    "total": lambda c: c["total"],
    "exact,ctc": lambda c: (c["exact"], c["ctc"]),
    "exact,ctc_ln": lambda c: (c["exact"], c["ctc_ln"]),
    "ctc,exact": lambda c: (c["ctc"], c["exact"]),
    "ctc_ln,exact": lambda c: (c["ctc_ln"], c["exact"]),
    "total,ctc_ln": lambda c: (c["total"], c["ctc_ln"]),
    "ctc_ln,total": lambda c: (c["ctc_ln"], c["total"]),
    "exact,total": lambda c: (c["exact"], c["total"]),
    "total,exact": lambda c: (c["total"], c["exact"]),
}


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
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), total=num(c, "total_score"),
                             ctc=raw, ctc_ln=ln))
        if not pool or not ref or not uid:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        cache.append(dict(uid=uid, old=old, ref=ref, sub=sub, hw=hw, keys=desig.get(uid, [])))

    chars = sum(len(c["ref"]) for c in cache)
    men = sum(len(c["keys"]) for c in cache)
    print("# %s   rows=%d chars=%d mentions=%d" % (a.label, len(cache), chars, men))
    print("%-24s %-10s %8s %9s %8s %8s" % ("key", "floor", "CER%", "dCER pp", "recall%", "drec pp"))

    def run(keyname, floor):
        kf = KEYS[keyname]
        e = rec = 0
        for c in cache:
            sub = c["sub"]
            if floor is not None:
                ok = [x for x in sub if len(x["text"]) >= len(c["old"]) + floor]
                if ok:
                    sub = ok
                else:
                    e += edit_distance(list(c["ref"]), list(c["old"]))
                    rec += sum(1 for k in c["keys"] if k in c["old"])
                    continue
            new = max(sub, key=kf)["text"]
            e += edit_distance(list(c["ref"]), list(new))
            rec += sum(1 for k in c["keys"] if k in new)
        return 100 * e / chars, 100 * rec / men

    base_cer, base_rec = run("total", None)  # total_score argmax == front-end top-1 by construction
    print("%-24s %-10s %8.4f %+9.4f %8.2f %+8.2f" % ("(identity)", "-", base_cer, 0.0, base_rec, 0.0))
    for keyname in KEYS:
        for floor in (None, 0):
            cer, rec = run(keyname, floor)
            print("%-24s %-10s %8.4f %+9.4f %8.2f %+8.2f" % (
                keyname, "none" if floor is None else "no-shorten", cer, cer - base_cer, rec, rec - base_rec))


if __name__ == "__main__":
    main()
