#!/usr/bin/env python3
"""Principled acoustic uncertainty for "is this sentence trustworthy?"  (option B)

WHY.  The gate is the bottleneck: nothing on disk predicts "the top-1 is wrong" well enough
(best AUC 0.782, TPR@FPR=5% only 24.5%, and the frontier needs ~85%).  One principled reason
is that the front end exports `asr_score` = the PER-TOKEN MEAN log-probability of a candidate.
A mean destroys exactly the signal we want: "this sentence is fine except two characters where
the model was very unsure" averages out.  The standard, textbook quantity in ASR confidence
estimation is the N-BEST POSTERIOR, which needs the SEQUENCE log-likelihood (the sum, not the
mean).  We can reconstruct it: total ~= mean * n_tokens.

Signals evaluated here, all defined quantities rather than hand-picked features:
  * posterior over candidates from the MEAN score       (what the front end effectively uses)
  * posterior from the reconstructed SUM score          (sequence likelihood -- the correction)
  * p(top-1), entropy, and top1-top2 margin of each posterior
  * candidate-set DISPERSION: mean pairwise edit distance, and normalized entropy
Everything is measured by the operable metric -- TPR at a fixed FPR -- plus AUC for reference,
and reported IN-DOMAIN (AISHELL dev) as a sanity check as well as out-of-domain (ST-CMDS).

    acoustic_uncertainty.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
SETS = [
    ("AISHELL dev (in-domain sanity)", R / "train_aishell_v1" / "eval_aishell.messages.jsonl"),
    ("ST-CMDS held-out (target)", R / "e2eSTCMDS_VA.messages.jsonl"),
]


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def softmax(xs):
    mx = max(xs)
    ex = [math.exp(x - mx) for x in xs]
    z = sum(ex)
    return [e / z for e in ex]


def entropy(p):
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def load(path):
    out = []
    for line in Path(path).open(encoding="utf-8"):
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
            continue
        feat = {}
        for c in ((inp.get("cbwhisper") or {}).get("candidates") or []):
            k = norm(c.get("text") or "")
            if k and k not in feat:
                feat[k] = c
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        y = 1 if min(eds) < eds[0] else 0                       # the label: top-1 is wrong
        means, sums = [], []
        for t in cands:
            c = feat.get(t)
            if not c or c.get("asr_score") is None:
                continue
            m = float(c["asr_score"])
            means.append(m)
            # sequence log-likelihood reconstructed from the mean: sum = mean * n_tokens.
            # len(text) is the token-count proxy (Chinese: ~1 token per character).
            sums.append(m * max(1, len(t)))
        if len(means) < 2:
            continue
        f = {}
        pm = softmax(means)
        ps = softmax(sums)
        f["mean-posterior: p(top1)"] = pm[0]
        f["mean-posterior: entropy"] = entropy(pm)
        f["sum-posterior: p(top1)"] = ps[0]
        f["sum-posterior: entropy"] = entropy(ps)
        ms = sorted(means, reverse=True)
        ss = sorted(sums, reverse=True)
        f["mean score margin (1st-2nd)"] = ms[0] - ms[1]
        f["sum score margin (1st-2nd)"] = ss[0] - ss[1]
        # candidate-set dispersion
        pair = [edit_distance(list(cands[i]), list(cands[j]))
                for i in range(len(cands)) for j in range(i + 1, len(cands))]
        f["dispersion: mean pairwise edit dist"] = sum(pair) / max(1, len(pair))
        f["dispersion: max pairwise edit dist"] = max(pair) if pair else 0
        out.append((y, f))
    return out


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    rank, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        ar = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            if pairs[k][1]:
                rank += ar
        i = j + 1
    return (rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def tpr_at_fpr(scores, labels, fpr):
    """higher score = more likely a decision row (we flip as needed)"""
    pos = sorted((s for s, y in zip(scores, labels) if y), reverse=True)
    neg = sorted((s for s, y in zip(scores, labels) if not y), reverse=True)
    if not pos or not neg:
        return float("nan"), float("nan")
    thr = neg[min(len(neg) - 1, int(fpr * len(neg)))]
    tp = sum(1 for s in pos if s > thr)
    fp = sum(1 for s in neg if s > thr)
    return 100.0 * tp / len(pos), (100.0 * tp / (tp + fp) if tp + fp else float("nan"))


for name, path in SETS:
    if not path.exists():
        print("%s MISSING" % name)
        continue
    rs = load(path)
    labels = [y for y, _ in rs]
    print("=" * 104)
    print("%s   rows=%d  decision rows=%d (%.1f%%)" % (name, len(rs), sum(labels), 100.0 * sum(labels) / len(rs)))
    print("=" * 104)
    names = sorted({k for _, f in rs for k in f})
    print("  %-40s %6s %7s %7s %8s %8s" % ("signal (valid direction)", "AUC", "TPR@1%", "TPR@2%", "TPR@5%", "prec@5%"))
    rows = []
    for n in names:
        v = [f.get(n, float("nan")) for _, f in rs]
        good = [x for x in v if x == x]
        if len(good) < len(v) * 0.5:
            continue
        mean = sum(good) / len(good)
        vv = [x if x == x else mean for x in v]
        a = auc(vv, labels)
        flip = a < 0.5
        use = [-x for x in vv] if flip else vv
        cells = [tpr_at_fpr(use, labels, f)[0] for f in (0.01, 0.02, 0.05)]
        _, p5 = tpr_at_fpr(use, labels, 0.05)
        rows.append((cells[2], n, a, cells, p5, flip))
    for _, n, a, cells, p5, flip in sorted(rows, reverse=True):
        print("  %-40s %6.3f %7.1f %7.1f %8.1f %8.1f%s"
              % (n, a, cells[0], cells[1], cells[2], p5, "  (flipped)" if flip else ""))
    print("  %s" % ("frontier target: TPR ~85% at FPR=5%" if "ST-CMDS" in name else
                    "in-domain sanity: a usable signal must do clearly better here"))
    print()
