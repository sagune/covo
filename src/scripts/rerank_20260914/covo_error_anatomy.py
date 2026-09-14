#!/usr/bin/env python3
"""Where exactly does the backend's remaining headroom sit?

For a given predictions file this classifies every utterance by what the backend did
relative to its input, and how much CER each behaviour contributes:

  no_change         output equals the front-end top-1
  sel_win           output is a pool candidate and strictly better than the top-1
  sel_loss          output is a pool candidate and strictly worse
  edit_win          output is NOT a pool candidate and better than the top-1
  edit_loss         output is NOT a pool candidate and worse
and, independently, how much CER is still on the table:

  miss_visible      a candidate in the visible n-best would have been better
  miss_pool         a candidate in the full pool would have been better
  ref_not_in_pool   the reference itself is unreachable from the pool (needs EDITING)

--policy selects which text is scored, so the "input -> output" line reproduces the
official number:
  raw          the bare 9B rewrite (r["prediction"])
  deployable   restore[deployable]: protected spans from input-side evidence only
"""
import argparse
import difflib
import json
import sys
from collections import defaultdict
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
    ap.add_argument("--policy", default="deployable", choices=("raw", "deployable"))
    a = ap.parse_args()

    cat = defaultdict(lambda: dict(n=0, e_in=0, e_out=0))
    miss = defaultdict(lambda: dict(n=0, gap=0))
    xtab = defaultdict(int)          # (behaviour, recoverable?) -> rows
    xtab_gap = defaultdict(int)      # (behaviour, recoverable?) -> chars
    chars = 0
    total_in = total_out = 0
    for l in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        inp_raw = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp_raw.get("asr_top1") or "")
        pred = norm(r.get("prediction") or "")
        if not ref or not top:
            continue
        visible = set(texts_of(inp_raw.get("nbest")))
        pool = visible | set(texts_of((inp_raw.get("cbwhisper") or {}).get("candidates")))
        terms = texts_of(inp_raw.get("prompt_hotwords")) or texts_of(inp_raw.get("hotwords"))
        dep = [t for t in terms if t in top and any(t in c for c in pool)]
        out = restore(top, pred, spans_in(top, dep)) if a.policy == "deployable" else pred

        e_in = edit_distance(list(ref), list(top))
        e_out = edit_distance(list(ref), list(out)) if out else e_in
        chars += len(ref)
        total_in += e_in
        total_out += e_out

        if out == top:
            k = "no_change"
        elif out in pool:
            k = "sel_win" if e_out < e_in else ("sel_loss" if e_out > e_in else "sel_tie")
        else:
            k = "edit_win" if e_out < e_in else ("edit_loss" if e_out > e_in else "edit_tie")
        c = cat[k]
        c["n"] += 1
        c["e_in"] += e_in
        c["e_out"] += e_out

        if visible:
            ev = min(edit_distance(list(ref), list(t)) for t in visible)
            if ev < e_out:
                miss["miss_visible"]["n"] += 1
                miss["miss_visible"]["gap"] += e_out - ev
                xtab[(k, "visible")] += 1
                xtab_gap[(k, "visible")] += e_out - ev
            else:
                xtab[(k, "none")] += 1
        if pool:
            ep = min(edit_distance(list(ref), list(t)) for t in pool)
            if ep < e_out:
                miss["miss_pool"]["n"] += 1
                miss["miss_pool"]["gap"] += e_out - ep
        if ref not in pool and e_out > 0:
            miss["ref_not_in_pool"]["n"] += 1

    print("# %s   policy=%s   chars=%d" % (a.label, a.policy, chars))
    print("  input  CER %.4f%%   ->   output CER %.4f%%   (net %+.4f pp)" % (
        100 * total_in / chars, 100 * total_out / chars, 100 * (total_out - total_in) / chars))
    print("  distance to 4.5%%: %+.4f pp (%+.0f chars)" % (
        100 * total_out / chars - 4.5, total_out - 4.5 * chars / 100))
    print()
    print("  %-12s %6s %8s %9s %9s %11s   %s" % (
        "behaviour", "rows", "%rows", "e_in", "e_out", "net edits", "rows w/ better visible cand (gap)"))
    order = ["no_change", "sel_win", "sel_tie", "sel_loss", "edit_win", "edit_tie", "edit_loss"]
    n_all = sum(c["n"] for c in cat.values())
    for k in order:
        c = cat.get(k)
        if not c:
            continue
        print("  %-12s %6d %7.1f%% %9d %9d %+11d   %5d (%d chars)" % (
            k, c["n"], 100.0 * c["n"] / max(1, n_all), c["e_in"], c["e_out"], c["e_out"] - c["e_in"],
            xtab[(k, "visible")], xtab_gap[(k, "visible")]))
    print()
    print("  remaining headroom (CER points over all chars):")
    for k in ("miss_visible", "miss_pool"):
        m = miss.get(k)
        if m:
            print("    %-16s %5d rows  %8.4f pp recoverable by a better SELECTOR" % (k, m["n"], 100.0 * m["gap"] / chars))
    m = miss.get("ref_not_in_pool")
    if m:
        print("    %-16s %5d rows  (reference unreachable from the pool -> needs EDITING)" % ("ref_not_in_pool", m["n"]))


if __name__ == "__main__":
    main()
