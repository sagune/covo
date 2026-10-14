#!/usr/bin/env python3
"""Generic version of P3': apply protect + forced-CTC reranking to any dataset."""
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
    out_rows, changed, skipped = [], 0, 0
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
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=scores[t]))
        if not pool:
            skipped += 1
            out_rows.append(r)
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        best = max(sub, key=lambda c: (c["exact"], c["ctc"]))

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
    print("  rows %d | top-1 changed %d (%.1f%%) | skipped %d" % (
        len(out_rows), changed, 100 * changed / len(out_rows), skipped))
    print("  front-end CER    : %.4f%% -> %.4f%%  (%+.4f pp)" % (
        100 * eb / chars, 100 * ea / chars, 100 * (eb - ea) / chars))
    if men:
        print("  designated recall: %.2f%% (%d/%d) -> %.2f%% (%d/%d)" % (
            100 * hb / men, hb, men, 100 * ha / men, ha, men))


if __name__ == "__main__":
    main()
