#!/usr/bin/env python3
"""Cross-dataset check: keep the corrector's work, restore protected spans.

Usage:
  restore_eval.py --records <predictions.jsonl> --label <name> [--aligned <aligned.txt>]

Policies
  raw                  deployed full-sentence rewrite
  verifier             covo.edits.validate_edits, reject -> revert whole sentence
  restore[deployable]  character-level restore of protected spans derived ONLY from
                       input-side evidence (prompt_hotwords/hotwords present in the
                       top-1 and corroborated by the candidate pool)
  restore[annotation]  same, but the protected set is the designated hotword
                       annotation (upper bound; not deployable)
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))

from covo.edits import Edit, apply_edits, validate_edits     # noqa: E402
from covo.text import normalize_chinese_text as norm         # noqa: E402
from covo.metrics import edit_distance                       # noqa: E402


def opcodes(a, b):
    return [x for x in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if x[0] != "equal"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--aligned", default=None)
    ap.add_argument("--uttid-file", default=None,
                    help="file of '<uttid> <transcript>' lines in evaluation row order, "
                         "for datasets whose records only carry a row index as id")
    args = ap.parse_args()

    desig = {}
    if args.aligned:
        for line in open(args.aligned, encoding="utf-8-sig"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                desig.setdefault(f[1].strip(), []).append(norm(f[0]))

    pos2utt = None
    if args.uttid_file:
        pos2utt = []
        for line in open(args.uttid_file, encoding="utf-8-sig"):
            p = line.rstrip("\n").split(maxsplit=1)
            if p:
                pos2utt.append((p[0], norm(p[1]) if len(p) > 1 else ""))

    rows = []
    with open(args.records, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    prep = []
    mism = 0
    for r in rows:
        raw_in = r.get("input") or {}
        inp, pred, ref = norm(raw_in.get("asr_top1") or ""), norm(r.get("prediction") or ""), norm(r.get("reference") or "")
        if not inp or not ref:
            continue
        uid = str(r.get("id"))
        if pos2utt is not None:
            try:
                idx = int(uid)
                cand_uid, cand_ref = pos2utt[idx]
                if cand_ref and cand_ref != ref:
                    mism += 1
                uid = cand_uid
            except (ValueError, IndexError):
                mism += 1
        pool = set(texts_of(raw_in.get("nbest"))) | set(texts_of((raw_in.get("cbwhisper") or {}).get("candidates")))
        terms = texts_of(raw_in.get("prompt_hotwords")) or texts_of(raw_in.get("hotwords"))
        deployable = [t for t in terms if t in inp and any(t in c for c in pool)]
        annotation = [k for k in desig.get(uid, []) if k in inp]
        prep.append(dict(uid=uid, inp=inp, pred=pred, ref=ref, pool=pool,
                         deployable=deployable, annotation=annotation,
                         desig=[norm(k) for k in desig.get(uid, [])],
                         evidence={"asr_top1": inp, "nbest": texts_of(raw_in.get("nbest")),
                                   "hotwords": [{"text": t} for t in texts_of(raw_in.get("hotwords"))]}))

    TOTAL = sum(len(p["ref"]) for p in prep)
    NCAND = sum(len(p["desig"]) for p in prep)
    print("# %s" % args.label)
    print("rows %d | reference chars %d | designated mentions %d | ref-in-pool %.1f%% | mean pool %.1f" % (
        len(prep), TOTAL, NCAND,
        100 * sum(p["ref"] in p["pool"] for p in prep) / len(prep),
        sum(len(p["pool"]) for p in prep) / len(prep)))
    if pos2utt is not None:
        print("row-order id mapping: %d/%d rows whose reference disagrees with the uttid file" % (mism, len(prep)))

    outs = {"input": {}, "raw": {}, "verifier": {}, "restore[deployable]": {}, "restore[annotation]": {}}
    for p in prep:
        i = p["uid"]
        outs["input"][i] = p["inp"]
        outs["raw"][i] = p["pred"]
        edits = opcodes(p["inp"], p["pred"])
        es = [Edit(from_text=p["inp"][i2:j2], to_text=p["pred"][k2:l2]) for _, i2, j2, k2, l2 in edits]
        ok, _ = validate_edits(es, p["evidence"], max_total_changed_chars=12)
        t, _, rej = apply_edits(p["inp"], es)
        outs["verifier"][i] = t if (ok and not rej) else p["inp"]
        outs["restore[deployable]"][i] = restore(p["inp"], p["pred"], spans_in(p["inp"], p["deployable"]))
        outs["restore[annotation]"][i] = restore(p["inp"], p["pred"], spans_in(p["inp"], p["annotation"]))

    for name, out in outs.items():
        chars = err = edited = hits = destroyed = 0
        for p in prep:
            t = out[p["uid"]]
            chars += len(p["ref"])
            err += edit_distance(list(p["ref"]), list(t))
            edited += t != p["inp"]
            for k in p["desig"]:
                hits += k in t
                destroyed += (k in p["inp"] and k not in t)
        sub = [p for p in prep if p["ref"] not in p["pool"]]
        c = sum(len(p["ref"]) for p in sub)
        ie = sum(edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
        oe = sum(edit_distance(list(p["ref"]), list(out[p["uid"]])) for p in sub)
        imp = sum(edit_distance(list(p["ref"]), list(out[p["uid"]])) <
                  edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
        wor = sum(edit_distance(list(p["ref"]), list(out[p["uid"]])) >
                  edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
        rtxt = "recall %5.2f%% (%d/%d) destroyed %3d" % (100 * hits / NCAND, hits, NCAND, destroyed) if NCAND else "recall n/a"
        print("  %-22s CER %7.4f%% | %s | edited %4d | beyond-N-best %+.3f pp (imp %d wor %d, n=%d)" % (
            name, 100 * err / chars, rtxt, edited, 100 * (ie - oe) / TOTAL, imp, wor, len(sub)))
    print()


if __name__ == "__main__":
    main()
