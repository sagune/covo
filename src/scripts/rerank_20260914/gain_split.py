#!/usr/bin/env python3
"""Split the backend's current gain into the part selection could have produced and the
part only editing could produce, and price the remaining headroom on each side.

This matters because the two are trained by the same objective but are different
capabilities, and the goal protects one (editing) while buying the other (selection).

  rows with    reference IN the pool      -> a perfect selector could reach the pool best
  rows with    reference NOT in the pool  -> only generation/editing can help
"""
import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402


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
    a = ap.parse_args()

    acc = {"in": dict(n=0, ch=0, e0=0, e1=0, poolbest=0, visbest=0),
           "out": dict(n=0, ch=0, e0=0, e1=0, poolbest=0, visbest=0)}
    for line in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        visible = set(texts_of(inp.get("nbest")))
        pool = visible | set(texts_of((inp.get("cbwhisper") or {}).get("candidates")))
        terms = texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords"))
        dep = [t for t in terms if t in top and any(t in c for c in pool)]
        out = restore(top, norm(r.get("prediction") or ""), spans_in(top, dep))
        e0 = edit_distance(list(ref), list(top))
        e1 = edit_distance(list(ref), list(out)) if out else e0
        ev = min((edit_distance(list(ref), list(t)) for t in visible), default=e0)
        ep = min((edit_distance(list(ref), list(t)) for t in pool), default=e0)
        k = "in" if ref in pool else "out"
        d = acc[k]
        d["n"] += 1
        d["ch"] += len(ref)
        d["e0"] += e0
        d["e1"] += e1
        d["poolbest"] += ep
        d["visbest"] += ev

    ins, outs = acc["in"], acc["out"]
    tot_ch = ins["ch"] + outs["ch"]
    tot_e0 = ins["e0"] + outs["e0"]
    tot_e1 = ins["e1"] + outs["e1"]
    print("# %s   (policy=restore[deployable])   total chars=%d" % (a.label, tot_ch))
    print("  whole set   input CER %7.4f%%  ->  output CER %7.4f%%   net %+.0f chars"
          % (100 * tot_e0 / tot_ch, 100 * tot_e1 / tot_ch, tot_e0 - tot_e1))
    print()
    print("  %-26s %6s %8s %9s %9s %10s %10s" % ("partition", "rows", "chars", "in CER", "out CER", "gain chars", "capture"))
    for name, d in (("ref IN pool (selection)", ins), ("ref NOT in pool (editing)", outs)):
        gain = d["e0"] - d["e1"]
        avail = d["e0"] - d["visbest"] if name.startswith("ref IN") else 0
        cap = ("%.1f%% of %d" % (100.0 * gain / avail, avail)) if avail else "n/a"
        print("  %-26s %6d %8d %8.4f%% %8.4f%% %+10d %10s"
              % (name, d["n"], d["ch"], 100 * d["e0"] / d["ch"], 100 * d["e1"] / d["ch"], gain, cap))
    print()
    gi = ins["e0"] - ins["e1"]
    go = outs["e0"] - outs["e1"]
    print("  gain from SELECTION : %+5d chars (%.0f%% of total gain)" % (gi, 100.0 * gi / max(1, gi + go)))
    print("  gain from EDITING   : %+5d chars (%.0f%% of total gain)" % (go, 100.0 * go / max(1, gi + go)))
    print("  selection headroom still on the table (visible-only oracle): %d chars = %.4f pp"
          % (ins["e0"] - ins["visbest"], 100.0 * (ins["e0"] - ins["visbest"]) / tot_ch))
    print("  full-pool oracle would be %.4f%% (%.4f pp below the current output)"
          % (100 * (ins["poolbest"] + outs["poolbest"]) / tot_ch,
             100.0 * (tot_e1 - ins["poolbest"] - outs["poolbest"]) / tot_ch))
    print("  distance to 4.5%%: %+.0f chars" % (tot_e1 - 4.5 * tot_ch / 100))


if __name__ == "__main__":
    main()
