#!/usr/bin/env python3
"""THE UNTESTED QUESTION: which signal predicts "the top-1 is wrong here" (the gate, k)?

Everything so far measured SELECTION (a): given that a better candidate exists, can a signal
find it?  Results 12-13: features 4.4%, trained generator 20.9%, zero-shot scorer 30.3%, a
trivial min-edit rule 43.5%.  Selection is therefore NOT the problem -- the frontier needs
only a>=36.9% if k>=0.95.

Nobody has measured the GATE: can any signal say, per row, "this one needs changing"?  Those
are different tasks -- a feature can be useless at ranking candidates yet excellent at telling
whether the top-1 is in trouble.  This computes the AUC of every available signal for the
binary label "this row has a strictly better candidate" (oracle label, for measurement only).

If some signal reaches AUC ~0.9, then a gate at k=0.95 is feasible, and paired with the
min-edit rule's 43.5% selection the 4.5 target becomes reachable.  If nothing does, the honest
conclusion is that ST-CMDS's remaining 0.44pp needs information nobody has.

    gate_diagnostic.py [--records ...] [--scorer ...]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

DEF = "/root/autodl-tmp/.dsh_checks/rerank/e2eSTCMDS_VA.messages.jsonl"
DEF_SCORER = "/root/autodl-tmp/.dsh_checks/rerank/train_aishell_v1/stcmds_likelihood.predictions.jsonl"


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def auc(scores, labels):
    """Rank-based AUC = P(score(pos) > score(neg)).  Ties count a half."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=DEF)
    ap.add_argument("--scorer", default=DEF_SCORER)
    a = ap.parse_args()

    scorer = {}
    sp = Path(a.scorer)
    if sp.exists():
        for line in sp.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = norm((r.get("input") or {}).get("asr_top1") or "") + "|" + norm(r.get("reference") or "")
            scorer[key] = r.get("candidate_scores") or []

    feats = {}
    labels = []
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
            labels.append(0)
            continue
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        labels.append(1 if min(eds) < eds[0] else 0)
        feat = {}
        for c in ((inp.get("cbwhisper") or {}).get("candidates") or []):
            k = norm(c.get("text") or "")
            if k and k not in feat:
                feat[k] = c
        t0 = feat.get(top, {})

        def g(k2, default=0.0):
            try:
                return float(t0.get(k2) or default)
            except Exception:
                return default

        asr = [float(feat[t]["asr_score"]) for t in cands if t in feat and feat[t].get("asr_score") is not None]
        tot = [float(feat[t]["total_score"]) for t in cands if t in feat and feat[t].get("total_score") is not None]
        near1 = sum(1 for i in range(1, len(cands))
                    if edit_distance(list(top), list(cands[i])) <= 1)
        dmin = min(edit_distance(list(top), list(cands[i])) for i in range(1, len(cands)))

        row = {
            "top1_asr_score": g("asr_score"),
            "top1_total_score": g("total_score"),
            "top1_exact_weighted": g("exact_weighted_score"),
            "top1_phonetic": g("phonetic_score"),
            "top1_hotword": g("hotword_score"),
            "top1_consensus": g("consensus_score"),
            "n_candidates": len(cands),
            "n_within_1_edit_of_top1": near1,
            "min_edit_to_any_candidate": dmin,
            "score_margin_asr": (max(asr) - min(asr)) if len(asr) > 1 else 0.0,
            "score_margin_total": (max(tot) - min(tot)) if len(tot) > 1 else 0.0,
            "top1_is_best_by_asr": 1.0 if (asr and max(asr) == g("asr_score")) else 0.0,
            "top1_is_best_by_total": 1.0 if (tot and max(tot) == g("total_score")) else 0.0,
        }
        sc = scorer.get(top + "|" + ref)
        if sc:
            ls = [float(c.get("lm_score") or 0.0) for c in sc]
            mx = max(ls)
            ex = [math.exp(s - mx) for s in ls]
            z = sum(ex)
            p = [e / z for e in ex]
            row["scorer_entropy"] = -sum(pi * math.log(pi + 1e-12) for pi in p)
            row["scorer_top1_prob"] = p[0]
            row["scorer_margin"] = (mx - sorted(ls)[-2]) if len(ls) > 1 else 0.0
        else:
            row["scorer_entropy"] = row["scorer_top1_prob"] = row["scorer_margin"] = float("nan")
        feats.setdefault(len(labels) - 1, row)

    # a missing feature is NaN for everyone in some rows -> fill with the column mean
    names = sorted({k for r in feats.values() for k in r})
    cols = {n: [] for n in names}
    for i in range(len(labels)):
        r = feats.get(i, {})
        for n in names:
            cols[n].append(r.get(n, float("nan")))
    print("rows %d | decision rows %d (%.1f%%)" % (len(labels), sum(labels), 100.0 * sum(labels) / len(labels)))
    print()
    print("  %-30s %8s %8s" % ("signal", "AUC", "AUC(flip)"))
    res = []
    for n in names:
        v = cols[n]
        good = [x for x in v if x == x]
        if len(good) < len(v) * 0.5:
            print("  %-30s %8s   (too many missing)" % (n, "-"))
            continue
        mean = sum(good) / len(good)
        vv = [x if x == x else mean for x in v]
        aa = auc(vv, labels)
        res.append((max(aa, 1 - aa), n, aa))
    for best, n, aa in sorted(res, reverse=True):
        print("  %-30s %8.3f %8.3f%s" % (n, aa, 1 - aa, "   <-- usable gate" if best >= 0.85 else ""))
    print()
    print("  AUC 0.5 = no information.  A gate needs high AUC AND high sensitivity at the")
    print("  abstention rate we can afford (k=0.95 means only 5% of good rows may be touched).")


if __name__ == "__main__":
    main()
