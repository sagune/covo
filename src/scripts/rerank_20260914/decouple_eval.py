#!/usr/bin/env python3
"""Portable CB-SenseVoice x COVO decoupling evaluation.

Works on any predictions JSONL whose records carry `input.asr_top1`,
`input.nbest`, optional `input.cbwhisper.candidates`, optional
`input.hotwords` / `input.prompt_hotwords`, plus `prediction` and `reference`.

Reports, for each policy:
  - corpus CER, edit rate, fallback rate
  - protected-term destruction rate (input-side, reference-blind)
  - attribution buckets: same-as-input / in-pool selection / out-of-pool generation
  - beyond-N-best subset (reference absent from the pool) -- the only part a
    reranker over this pool could never reach

Policies:
  raw              deployed behaviour, no verifier
  verifier         covo.edits.validate_edits, reject -> revert whole utterance
  verifier+zone    same, but drop only the edits touching an evidence-supported
                   protected span, then re-validate the remainder
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))

from covo.edits import Edit, apply_edits, validate_edits   # noqa: E402
from covo.text import normalize_chinese_text as norm       # noqa: E402
from covo.metrics import edit_distance                      # noqa: E402

CHAR_CAP = 12


def texts_of(value):
    out = []
    for x in value or []:
        if isinstance(x, str):
            out.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            out.append(norm(x["text"]))
    return [t for t in out if t]


def diff_edits(a, b):
    out = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((Edit(from_text=a[i:j], to_text=b[k:l]), i, j))
    return out


def overlap(item, zone):
    _, i, j = item
    return any(i < z1 and j > z0 for z0, z1 in zone)


def zones(inp, protected, pool):
    """No-edit spans: protected terms present in top-1 that a candidate corroborates."""
    z = []
    for h in protected:
        if len(h) < 2 or h not in inp:
            continue
        if not any(h in c for c in pool):
            continue
        s = 0
        while True:
            p = inp.find(h, s)
            if p < 0:
                break
            z.append((p, p + len(h)))
            s = p + 1
    return z


def apply_policy(kind, inp, pred, pool, protected, evidence):
    edits = diff_edits(inp, pred)
    if kind == "raw":
        return pred, edits
    if kind == "verifier":
        ok, _ = validate_edits([e for e, _, _ in edits], evidence, max_total_changed_chars=CHAR_CAP)
        t, _, rej = apply_edits(inp, [e for e, _, _ in edits])
        return (t if (ok and not rej) else inp), edits
    z = zones(inp, protected, pool)
    keep = [e for e in edits if not overlap(e, z)]
    ok, _ = validate_edits([e for e, _, _ in keep], evidence, max_total_changed_chars=CHAR_CAP)
    t, _, rej = apply_edits(inp, [e for e, _, _ in keep])
    return (t if (ok and not rej) else inp), edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--aligned", default=None, help="AISHELL-style hotword->uttid file for designated recall")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    desig = {}
    if args.aligned:
        for line in open(args.aligned, encoding="utf-8-sig"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                desig.setdefault(f[1].strip(), []).append(norm(f[0]))

    rows = []
    with open(args.records, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_rows:
        rows = rows[:args.max_rows]

    prepared = []
    for r in rows:
        inp_raw = r.get("input") or {}
        inp = norm(inp_raw.get("asr_top1") or "")
        pred = norm(r.get("prediction") or "")
        ref = norm(r.get("reference") or "")
        if not inp or not ref:
            continue
        nbest = texts_of(inp_raw.get("nbest"))
        cands = texts_of((inp_raw.get("cbwhisper") or {}).get("candidates"))
        pool = set(nbest) | set(cands)
        protected = texts_of(inp_raw.get("prompt_hotwords")) or texts_of(inp_raw.get("hotwords"))
        evidence = {"asr_top1": inp, "nbest": nbest,
                    "hotwords": [{"text": t} for t in texts_of(inp_raw.get("hotwords"))]}
        prepared.append(dict(id=r.get("id"), inp=inp, pred=pred, ref=ref, pool=pool,
                             protected=protected, evidence=evidence,
                             desig=[norm(k) for k in desig.get(r.get("id"), [])]))

    total_chars = sum(len(p["ref"]) for p in prepared)
    print("# %s" % args.label)
    print("rows %d | reference chars %d | mean pool %d | ref-in-pool %.1f%%" % (
        len(prepared), total_chars,
        sum(len(p["pool"]) for p in prepared) / len(prepared),
        100 * sum(p["ref"] in p["pool"] for p in prepared) / len(prepared)))
    zone_rows = sum(bool(zones(p["inp"], p["protected"], p["pool"])) for p in prepared)
    print("rows with an evidence-supported protected span: %d (%.1f%%)" % (
        zone_rows, 100 * zone_rows / len(prepared)))
    print()

    summary = {}
    for kind in ("raw", "verifier", "verifier+zone"):
        out = {}
        for p in prepared:
            t, edits = apply_policy(kind, p["inp"], p["pred"], p["pool"], p["protected"], p["evidence"])
            out[p["id"]] = t
        chars = err = edited = destroyed = prot_tot = hits = tot = 0
        buckets = {"same-as-input": [], "in-pool selection": [], "out-of-pool generation": []}
        for p in prepared:
            t = out[p["id"]]
            chars += len(p["ref"])
            err += edit_distance(list(p["ref"]), list(t))
            edited += t != p["inp"]
            for k in p["desig"]:
                tot += 1
                hits += k in t
            for k in p["protected"]:
                if k in p["inp"]:
                    prot_tot += 1
                    destroyed += k not in t
            key = ("same-as-input" if t == p["inp"]
                   else "in-pool selection" if t in p["pool"]
                   else "out-of-pool generation")
            buckets[key].append(p)

        print("## %s" % kind)
        print("   CER %.4f%% | edit rate %.1f%% | protected destroyed %d/%d (%.1f%%)%s" % (
            100 * err / chars, 100 * edited / len(prepared), destroyed, prot_tot,
            100 * destroyed / max(prot_tot, 1),
            " | designated recall %.2f%% (%d/%d)" % (100 * hits / tot, hits, tot) if tot else ""))
        for key, ps in buckets.items():
            if not ps:
                continue
            c = sum(len(p["ref"]) for p in ps)
            ie = sum(edit_distance(list(p["ref"]), list(p["inp"])) for p in ps)
            oe = sum(edit_distance(list(p["ref"]), list(out[p["id"]])) for p in ps)
            print("     %-24s rows %4d | in CER %7.3f%% | out CER %7.3f%% | gain %+7.3f pp" % (
                key, len(ps), 100 * ie / c, 100 * oe / c, 100 * (ie - oe) / total_chars))

        # beyond-N-best subset
        sub = [p for p in prepared if p["ref"] not in p["pool"]]
        if sub:
            c = sum(len(p["ref"]) for p in sub)
            ie = sum(edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
            oe = sum(edit_distance(list(p["ref"]), list(out[p["id"]])) for p in sub)
            imp = sum(edit_distance(list(p["ref"]), list(out[p["id"]])) <
                      edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
            wor = sum(edit_distance(list(p["ref"]), list(out[p["id"]])) >
                      edit_distance(list(p["ref"]), list(p["inp"])) for p in sub)
            fixed = sum(out[p["id"]] == p["ref"] for p in sub)
            print("     beyond-N-best (ref not in pool): rows %d | in CER %.3f%% -> out CER %.3f%% | gain %+.3f pp | imp %d wor %d | exact-fixed %d" % (
                len(sub), 100 * ie / c, 100 * oe / c, 100 * (ie - oe) / total_chars, imp, wor, fixed))
        print()
        summary[kind] = dict(cer=err / chars, edit_rate=edited / len(prepared),
                             protected_destroyed=destroyed, protected_total=prot_tot,
                             designated_recall=(hits / tot if tot else None),
                             buckets={k: len(v) for k, v in buckets.items()})

    if args.out:
        Path(args.out).write_text(json.dumps({args.label: summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
