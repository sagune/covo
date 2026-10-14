#!/usr/bin/env python3
"""Paired comparison of two COVO prediction files (arm A vs arm B).

bootstrap_two.py compares the front end (`input.asr_top1`) of two files, which is the
wrong quantity once the backend is the thing under test.  This compares the *outputs*
of two arms on the same message file and answers three questions at once:

  1. headline      restore[deployable] CER / recall / destroyed / edited for each arm,
                   plus the beyond-N-best (editing-ability) partition
  2. significance  paired bootstrap over utterances for dCER / drecall and the fraction
                   of resamples where B is no worse on both axes
  3. mechanism     a row-level behaviour migration matrix.  The bet behind the training
                   run is that rows the old adapter left alone (no_change) become
                   sel_win/edit_win.  This counts exactly that, and the reverse
                   regression (rows it used to win that it now loses).

    compare_arms.py --a <preds.jsonl> --b <preds.jsonl> --aligned <aligned.txt> \
                    --uttid <uttid> --label-a "existing" --label-b "trained"
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

import difflib


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


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--policy", default="deployable", choices=("raw", "deployable"))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(args.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = defaultdict(list)
    for line in Path(args.aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig[f[1].strip()].append(norm(f[0]))

    A, B = load(args.a), load(args.b)
    if len(A) != len(B):
        print("row count differs: %d vs %d" % (len(A), len(B)))
    n = min(len(A), len(B))

    rows = []
    top_mismatch = 0
    for i in range(n):
        a, b = A[i], B[i]
        ia, ib = a.get("input") or {}, b.get("input") or {}
        ref = norm(a.get("reference") or b.get("reference") or "")
        top = norm(ia.get("asr_top1") or "")
        if top != norm(ib.get("asr_top1") or ""):
            top_mismatch += 1
        if not ref or not top:
            continue
        uid = str(a.get("id"))
        if uid.isdigit() and int(uid) < len(pos2utt):
            uid = pos2utt[int(uid)]
        visible = set(texts_of(ia.get("nbest"))) | set(texts_of(ib.get("nbest")))
        pool = visible | set(texts_of((ia.get("cbwhisper") or {}).get("candidates"))) \
                        | set(texts_of((ib.get("cbwhisper") or {}).get("candidates")))
        terms = texts_of(ia.get("prompt_hotwords")) or texts_of(ia.get("hotwords"))
        dep = [t for t in terms if t in top and any(t in c for c in pool)]

        def finalize(rec):
            if args.policy == "raw":
                return norm(rec.get("prediction") or "")
            return restore(top, norm(rec.get("prediction") or ""), spans_in(top, dep))

        ta, tb = finalize(a), finalize(b)
        ea = edit_distance(list(ref), list(ta)) if ta else edit_distance(list(ref), list(top))
        eb = edit_distance(list(ref), list(tb)) if tb else edit_distance(list(ref), list(top))
        e0 = edit_distance(list(ref), list(top))

        def cls(t, e):
            if t == top:
                return "no_change"
            if t in pool:
                return "sel_win" if e < e0 else ("sel_loss" if e > e0 else "sel_tie")
            return "edit_win" if e < e0 else ("edit_loss" if e > e0 else "edit_tie")

        keys = desig.get(uid, [])
        rows.append(dict(
            chars=len(ref), e0=e0, ea=ea, eb=eb, ca=cls(ta, ea), cb=cls(tb, eb),
            m=len(keys),
            ra=sum(1 for x in keys if x in ta), rb=sum(1 for x in keys if x in tb),
            da=sum(1 for x in keys if x in top and x not in ta),
            db=sum(1 for x in keys if x in top and x not in tb),
            ta=ta, tb=tb, ref=ref, pool=pool, beyond=(ref not in pool),
        ))

    if not rows:
        print("no comparable rows")
        return
    C = sum(r["chars"] for r in rows)

    def agg(sample):
        c = sum(x["chars"] for x in sample)
        m = sum(x["m"] for x in sample)
        return (100 * sum(x["ea"] for x in sample) / c,
                100 * sum(x["eb"] for x in sample) / c,
                100 * sum(x["da"] for x in sample) / m if m else 0.0,
                100 * sum(x["db"] for x in sample) / m if m else 0.0,
                100 * sum(x["ra"] for x in sample) / m if m else 0.0,
                100 * sum(x["rb"] for x in sample) / m if m else 0.0)

    ca, cb, da, db, ra, rb = agg(rows)
    print("# %s  vs  %s      policy=%s   rows=%d chars=%d top1-mismatch=%d"
          % (args.label_a, args.label_b, args.policy, len(rows), C, top_mismatch))
    print("  %-22s CER %7.4f%%   destroyed %3d   recall %5.2f%%"
          % (args.label_a, ca, sum(x["da"] for x in rows), ra))
    print("  %-22s CER %7.4f%%   destroyed %3d   recall %5.2f%%"
          % (args.label_b, cb, sum(x["db"] for x in rows), rb))
    print("  delta B-A            CER %+.4f pp            recall %+.2f pp"
          % (cb - ca, rb - ra))

    for nm, key in (("A", "ea"), ("B", "eb")):
        sub = [r for r in rows if r["beyond"]]
        if sub:
            c = sum(x["chars"] for x in sub)
            i0 = sum(x["e0"] for x in sub)
            e = sum(x[key] for x in sub)
            imp = sum(1 for x in sub if x[key] < x["e0"])
            wor = sum(1 for x in sub if x[key] > x["e0"])
            print("  beyond-N-best %s       CER %7.4f%% -> %7.4f%%  %+.3f pp (imp %d wor %d, n=%d)"
                  % (nm, 100 * i0 / c, 100 * e / c, 100 * (i0 - e) / c, imp, wor, len(sub)))

    rng = random.Random(args.seed)
    dc, dr, both = [], [], 0
    ded, ed_better = [], 0
    N = len(rows)

    def agg_edit(sample):
        """Beyond-N-best partition: the rows where only generation can help, i.e. the
        editing-ability dashboard.  Returns net gain in CER points for A and B."""
        sub = [x for x in sample if x["beyond"]]
        if not sub:
            return 0.0, 0.0
        c = sum(x["chars"] for x in sub)
        return (100.0 * sum(x["e0"] - x["ea"] for x in sub) / c,
                100.0 * sum(x["e0"] - x["eb"] for x in sub) / c)

    for _ in range(args.draws):
        s = [rows[rng.randrange(N)] for _ in range(N)]
        x = agg(s)
        d = x[1] - x[0]
        e = x[5] - x[4]
        dc.append(d)
        dr.append(e)
        if d <= 0 and e >= 0:
            both += 1
        ea, eb = agg_edit(s)
        ded.append(eb - ea)
        if eb >= ea:
            ed_better += 1
    dc.sort(); dr.sort(); ded.sort()
    print("  paired bootstrap (%d draws): dCER %+.4f pp [%+.4f, %+.4f]   drecall %+.2f pp [%+.2f, %+.2f]"
          % (args.draws, cb - ca, dc[int(.025 * args.draws)], dc[int(.975 * args.draws)],
             rb - ra, dr[int(.025 * args.draws)], dr[int(.975 * args.draws)]))
    print("  P(B no worse on both axes) = %.3f" % (both / args.draws))
    print("  EDITING-ABILITY GATE (beyond-N-best partition):  d(edit gain) %+.4f pp "
          "[%+.4f, %+.4f]   P(B's editing >= A's) = %.3f"
          % (ded[len(ded) // 2], ded[int(.025 * args.draws)], ded[int(.975 * args.draws)],
             ed_better / args.draws))
    if ded[int(.025 * args.draws)] < -0.5:
        print("  ^^ WARNING: the lower bound says B may have LOST more than 0.5 pp of editing"
              " ability - this is the rollback condition the plan declares")

    mig = Counter((r["ca"], r["cb"]) for r in rows)
    cls_order = ["no_change", "sel_win", "sel_tie", "sel_loss", "edit_win", "edit_tie", "edit_loss"]
    present = [c for c in cls_order if any(c in k for k in mig)]
    print()
    print("  behaviour migration (rows)   rows: A -> B")
    print("  %-12s %s" % ("A \\ B", "".join("%11s" % c for c in present)))
    for x in present:
        print("  %-12s %s" % (x, "".join("%11d" % mig.get((x, y), 0) for y in present)))

    gained = sum(mig.get(("no_change", y), 0) for y in ("sel_win", "edit_win"))
    lost = sum(mig.get((x, "no_change"), 0) for x in ("sel_win", "edit_win"))
    net_a = sum((r["e0"] - r["ea"]) for r in rows)
    net_b = sum((r["e0"] - r["eb"]) for r in rows)
    print()
    print("  THE BET:  no_change rows that B turns into a win : %d" % gained)
    print("  REGRESSION: win rows that B turns back to no_change: %d" % lost)
    print("  chars saved vs the front end:  A %+d   B %+d   (B-A %+d)" % (net_a, net_b, net_b - net_a))
    flips = [r for r in rows if r["eb"] != r["ea"]]
    print("  rows where B differs from A at all: %d   of which better %d / worse %d"
          % (len(flips), sum(1 for r in flips if r["eb"] < r["ea"]), sum(1 for r in flips if r["eb"] > r["ea"])))
    if gained or lost:
        for r in [x for x in rows if x["ca"] == "no_change" and x["cb"] in ("sel_win", "edit_win")][:3]:
            print("    e.g. [%s->%s] ref=%s" % (r["ca"], r["cb"], r["ref"][:34]))
            print("         top=%s" % r["ta"][:44])
            print("         new=%s" % r["tb"][:44])


if __name__ == "__main__":
    main()
