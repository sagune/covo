#!/usr/bin/env python3
"""DECISIVE DIAGNOSTIC: on the rows where a better candidate exists, what signal can
actually find it -- acoustic/feature evidence that the front end already has, or the text?

Why this is the right question.  ST-CMDS sits at 4.9356% and the target is 4.5%.  Of the
5130 held-out utterances, 1461 (28.5%) have a candidate in the prompt's numbered list that
is strictly closer to the reference than the top-1, and the front end chose the top-1 on
every one of them by construction.  So EVERY point of available gain comes from picking
correctly on those 1461 rows.  The oracle over them is worth 2.7046 pp (4.9356 -> 2.2310),
i.e. the target only needs ~16% of it.

The question that decides the whole line of work is: which signal can do that picking?
  * the front end's own features (forced-CTC / phonetic / hotword coverage / total score)
    -- if these are the answer, the fix is FRONT-END, not a trained text model
  * the text alone -- if this is the answer, training text models was the right bet
  * neither -- then the target is not reachable and we should say so

Reports, for each signal, the top-1 accuracy ON the 1461 rows and the resulting corpus CER
if that signal's choice were adopted on them (top-1 kept everywhere else).  Oracle gives the
ceiling, so every arm is directly comparable to the 4.5 target.

Also tests whether `asr_score` is length-normalised: if it were an un-normalised total, it
would correlate strongly with candidate length, which is the length bias of the E38 audit.

    diagnostic_what_can_pick.py [--records ...] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

DEF = "/root/autodl-tmp/.dsh_checks/rerank/e2eSTCMDS_VA.messages.jsonl"


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=DEF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--predictions", default="",
                    help="an arm's predictions file; reports its accuracy ON the decision rows "
                         "and its damage on the non-decision rows separately")
    a = ap.parse_args()

    rows = []
    for line in Path(a.records).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        cands = []
        seen = set()
        for t in [top] + texts_of(inp.get("nbest")):
            if t and t not in seen:
                seen.add(t)
                cands.append(t)
        feat = {}
        for c in ((inp.get("cbwhisper") or {}).get("candidates") or []):
            k = norm(c.get("text") or "")
            if k and k not in feat:
                feat[k] = c
        if len(cands) < 2:
            continue
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        e0 = eds[0]
        if min(eds) >= e0:
            continue                       # no better candidate visible -> not a decision row
        best_i = min(range(len(cands)), key=lambda i: (eds[i], i))
        rows.append(dict(ref=ref, cands=cands, eds=eds, best=best_i, feat=feat))
        if a.limit and len(rows) >= a.limit:
            break

    TOTAL_CH = 0
    BASE_ERR = 0
    all_rows = 0
    for line in Path(a.records).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        all_rows += 1
        TOTAL_CH += len(ref)
        BASE_ERR += edit_distance(list(ref), list(top))

    print("=" * 100)
    print("POPULATION")
    print("=" * 100)
    print("  corpus rows            : %d   chars %d" % (all_rows, TOTAL_CH))
    print("  front-end (top-1) CER  : %.4f%%" % (100.0 * BASE_ERR / TOTAL_CH))
    print("  DECISION ROWS (a strictly better candidate is visible): %d (%.1f%%)"
          % (len(rows), 100.0 * len(rows) / all_rows))
    gain = sum(r["eds"][0] - r["eds"][r["best"]] for r in rows)
    print("  oracle over those rows : %.4f pp  (%.4f%% -> %.4f%%)   target needs %.4f pp of it"
          % (100.0 * gain / TOTAL_CH, 100.0 * BASE_ERR / TOTAL_CH,
             100.0 * (BASE_ERR - gain) / TOTAL_CH, 100.0 * gain / TOTAL_CH - (100.0 * BASE_ERR / TOTAL_CH - 4.5)))
    print()

    def arm(label, pick):
        """pick(row) -> candidate index; score accuracy on decision rows and corpus CER."""
        correct = 0
        err = 0
        for r in rows:
            i = pick(r)
            if i == r["best"]:
                correct += 1
            err += r["eds"][i]
        cer = 100.0 * (BASE_ERR - sum(r["eds"][0] for r in rows) + err) / TOTAL_CH
        print("  %-34s top1-acc %5.1f%%   corpus CER %7.4f%%   %s"
              % (label, 100.0 * correct / len(rows), cer,
                 "<= TARGET" if cer <= 4.5 else ""))
        return cer

    def feat_pick(key, scale=1.0):
        def f(r):
            best_i, best_v = 0, None
            for i, t in enumerate(r["cands"]):
                c = r["feat"].get(t)
                if not c:
                    continue
                v = float(c.get(key) or 0.0) * scale
                if best_v is None or v > best_v:
                    best_v, best_i = v, i
            return best_i
        return f

    # Optional: an actual arm's predictions, scored separately on decision vs non-decision rows.
    # This is the TEXT side of the question.  A fine-tuned text model is an upper-ish bound on
    # what text alone can do here, and splitting its behaviour shows whether its failure is
    # "cannot pick" (low accuracy on decision rows) or "picks but will not abstain" (good
    # accuracy on decision rows, damage elsewhere).
    if a.predictions and Path(a.predictions).exists():
        preds = []
        for line in Path(a.predictions).open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                preds.append(norm(r.get("prediction") or ""))
        dec = {id(r): None for r in rows}
        # rebuild per-row index: the decision rows were collected in file order, so recompute
        di = 0
        hit = miss = 0
        err_dec = 0.0
        err_non = 0.0
        err_non_base = 0.0
        ch_non = 0
        pi = 0
        okrows = 0
        for line in Path(a.records).open(encoding="utf-8"):
            if not line.strip():
                continue
            rr = json.loads(line)
            inp = rr.get("input") or {}
            ref = norm(rr.get("reference") or "")
            top = norm(inp.get("asr_top1") or "")
            if not ref or not top:
                continue
            pred = preds[pi] if pi < len(preds) else top
            pi += 1
            cands, seen = [], set()
            for t in [top] + texts_of(inp.get("nbest")):
                if t and t not in seen:
                    seen.add(t)
                    cands.append(t)
            if len(cands) < 2:
                continue
            eds = [edit_distance(list(ref), list(t)) for t in cands]
            if min(eds) >= eds[0]:
                ch_non += len(ref)
                err_non += edit_distance(list(ref), list(pred))
                err_non_base += eds[0]
            else:
                okrows += 1
                best_i = min(range(len(cands)), key=lambda i: (eds[i], i))
                if pred == cands[best_i]:
                    hit += 1
                else:
                    miss += 1
                err_dec += edit_distance(list(ref), list(pred))
        print()
        print("=" * 100)
        print("TEXT ARM (a real model's output, split by row type): %s" % Path(a.predictions).name)
        print("=" * 100)
        if okrows:
            print("  ON the %d decision rows : best-candidate accuracy %.1f%%  (%d hit / %d miss)"
                  % (okrows, 100.0 * hit / okrows, hit, miss))
            base_dec = sum(r["eds"][0] for r in rows)
            print("     its errors there %d vs the front end's %d  ->  %+.0f chars"
                  % (err_dec, base_dec, err_dec - base_dec))
        if ch_non:
            print("  ON the %d non-decision chars : errors %d vs the front end's %d  ->  %+.0f chars"
                  % (ch_non, err_non, err_non_base, err_non - err_non_base))
            print("     ^ this is the damage from editing rows where no better candidate existed")

    print()
    print("=" * 100)
    print("ARMS  (accuracy on the %d decision rows; corpus CER if adopted there only)" % len(rows))
    print("=" * 100)
    arm("do nothing (front-end top-1)", lambda r: 0)
    arm("best forced-CTC asr_score", feat_pick("asr_score"))
    arm("best total_score (front end)", feat_pick("total_score"))
    arm("best exact_weighted_score", feat_pick("exact_weighted_score"))
    arm("best phonetic_score", feat_pick("phonetic_score"))
    arm("best hotword_score", feat_pick("hotword_score"))

    # Trivial heuristics: without these, an accuracy figure has no scale.  "most similar to
    # top-1" matters most -- on these rows the best candidate is usually a small edit of the
    # top-1, so a text model that only scores well because it prefers near-copies proves little.
    def sim_to_top1(r):
        best_i, best_v = 1, None
        for i in range(1, len(r["cands"])):
            v = -edit_distance(list(r["cands"][0]), list(r["cands"][i]))
            if best_v is None or v > best_v:
                best_v, best_i = v, i
        return best_i

    arm("most similar to top-1 (heuristic)", sim_to_top1)
    arm("longest candidate (heuristic)", lambda r: max(range(len(r["cands"])), key=lambda i: (len(r["cands"][i]), -i)))
    arm("shortest candidate (heuristic)", lambda r: min(range(len(r["cands"])), key=lambda i: (len(r["cands"][i]), i)))
    arm("ORACLE (upper bound)", lambda r: r["best"])

    print()
    print("=" * 100)
    print("DEPLOYABLE CHECK: apply the rule to EVERY row.  The arms above replace only the")
    print("decision rows, which needs oracle knowledge of which rows those are -- not a")
    print("deployable rule.  This applies it everywhere, so damage on non-decision rows counts.")
    print("=" * 100)

    def global_arm(label, pick):
        err = 0
        changed = 0
        for line in Path(a.records).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            inp = r.get("input") or {}
            ref = norm(r.get("reference") or "")
            top = norm(inp.get("asr_top1") or "")
            if not ref or not top:
                continue
            cands, seen = [], set()
            for t in [top] + texts_of(inp.get("nbest")):
                if t and t not in seen:
                    seen.add(t)
                    cands.append(t)
            if len(cands) < 2:
                err += edit_distance(list(ref), list(top))
                continue
            eds = [edit_distance(list(ref), list(t)) for t in cands]
            i = pick(dict(ref=ref, cands=cands, eds=eds))
            if i != 0:
                changed += 1
            err += eds[i]
        cer = 100.0 * err / TOTAL_CH
        print("  %-40s corpus CER %7.4f%%   changed %5d rows   %s"
              % (label, cer, changed, "<= TARGET" if cer <= 4.5 else ""))
        return cer

    global_arm("min-edit-to-top1 rule, ALL rows", sim_to_top1)
    global_arm("do nothing (top-1 everywhere)", lambda r: 0)

    # Gate sweep: act only when the chosen correction is SMALL.  This is deployable (it uses
    # only the candidate list and the top-1, never the reference) and it is the natural way to
    # buy restraint: a 1-character correction is far more likely to be right than a rewrite.
    print()
    print("  --- gate: apply the min-edit rule only if it changes at most K characters ---")
    for K in (1, 2, 3, 4):
        def gated(r, K=K):
            i = sim_to_top1(r)
            d = edit_distance(list(r["cands"][0]), list(r["cands"][i]))
            return i if d <= K else 0
        global_arm("  K<=%d" % K, gated)
    print()
    print("  (every arm keeps the front-end top-1 outside the decision rows, so the")
    print("   'do nothing' row reproduces the 4.9356%% baseline exactly)")

    # length-normalisation sanity check on asr_score
    import statistics
    xs, ys = [], []
    for r in rows:
        for t in r["cands"]:
            c = r["feat"].get(t)
            if c and c.get("asr_score") is not None:
                xs.append(len(t))
                ys.append(float(c["asr_score"]))
    if len(xs) > 10:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        print()
        print("  asr_score vs candidate length: pearson r = %+.3f  (n=%d)"
              % (cov / (sx * sy), len(xs)))
        print("  interpretation: |r| near 1 => an UN-normalised total (E38 length bias);"
              " near 0 => already a per-token mean, safe to use raw.")


if __name__ == "__main__":
    main()
