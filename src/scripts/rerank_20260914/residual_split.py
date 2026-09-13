#!/usr/bin/env python3
"""Where does the remaining error mass live?

Splits each dataset by whether the reference is reachable from the front-end's
candidate pool, then reports row/char/error shares and the pool oracle. This is
the measurement that decides what a third module would have to attack.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))
from covo.text import normalize_chinese_text as norm     # noqa: E402
from covo.metrics import edit_distance                    # noqa: E402


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def spans_in(text, terms):
    z = []
    for t in terms:
        if len(t) < 2:
            continue
        s = 0
        while True:
            p = text.find(t, s)
            if p < 0:
                break
            z.append((p, p + len(t)))
            s = p + 1
    return z


def restore(inp, pred, zone):
    if not zone:
        return pred
    out = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, inp, pred, autojunk=False).get_opcodes():
        if tag != "equal" and any(i < z1 and j > z0 for z0, z1 in zone):
            out.append(inp[i:j])
        else:
            out.append(pred[k:l])
    return "".join(out)


def load(records):
    prep = []
    with open(records, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            raw_in = r.get("input") or {}
            inp = norm(raw_in.get("asr_top1") or "")
            pred = norm(r.get("prediction") or "")
            ref = norm(r.get("reference") or "")
            if not inp or not ref:
                continue
            pool = set(texts_of(raw_in.get("nbest"))) | set(texts_of((raw_in.get("cbwhisper") or {}).get("candidates")))
            terms = texts_of(raw_in.get("prompt_hotwords")) or texts_of(raw_in.get("hotwords"))
            dep = [t for t in terms if t in inp and any(t in c for c in pool)]
            fixed = restore(inp, pred, spans_in(inp, dep))
            oracle = min((edit_distance(list(ref), list(c)) for c in pool), default=None)
            prep.append(dict(inp=inp, pred=pred, ref=ref, fixed=fixed, pool=pool, oracle=oracle))
    return prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    prep = load(args.records)
    total_chars = sum(len(p["ref"]) for p in prep)
    in_pool = [p for p in prep if p["ref"] in p["pool"]]
    out_pool = [p for p in prep if p["ref"] not in p["pool"]]

    def errs(rows, key):
        return sum(edit_distance(list(p["ref"]), list(p[key])) for p in rows)

    base_err = errs(prep, "inp")
    fixed_err = errs(prep, "fixed")
    oracle_err = sum(p["oracle"] for p in prep if p["oracle"] is not None)

    print("# %s" % args.label)
    print("rows %d | chars %d | input edits %d | restored edits %d | pool-oracle edits %d" % (
        len(prep), total_chars, base_err, fixed_err, oracle_err))
    print()
    print("%-22s %6s %7s %9s %9s %9s %9s %9s" % (
        "subset", "rows", "char%", "input CER", "restored", "oracle", "err% in", "err% rest"))
    for name, rows in (("reference IN pool", in_pool), ("reference NOT in pool", out_pool)):
        if not rows:
            continue
        c = sum(len(p["ref"]) for p in rows)
        ie = errs(rows, "inp")
        fe = errs(rows, "fixed")
        oe = sum(p["oracle"] for p in rows if p["oracle"] is not None)
        print("%-22s %6d %6.1f%% %8.3f%% %8.3f%% %8.3f%% %8.1f%% %8.1f%%" % (
            name, len(rows), 100 * c / total_chars, 100 * ie / c, 100 * fe / c, 100 * oe / c,
            100 * ie / max(base_err, 1), 100 * fe / max(fixed_err, 1)))
    print()
    print("  -> the %.1f%% of rows that the pool cannot reach carry %.1f%% of the remaining errors after restoration" % (
        100 * len(out_pool) / len(prep),
        100 * errs(out_pool, "fixed") / max(fixed_err, 1)))
    print()


if __name__ == "__main__":
    main()
