#!/usr/bin/env python3
"""Idea A pilot: can a *learned* candidate ranker beat the hand-designed rule?

Protocol: grouped 5-fold cross-validation over utterances on AISHELL dev.  For every
(utterance, candidate) pair we build the features that are already logged in the
evidence plus the forced-CTC score, and label the pool's minimum-CER candidate(s).
The label needs the reference, which is legitimate at training time only; at
inference the model sees exactly the same inputs as the hand rule.

Reported on the held-out fold, all on the same rows and the same pool:
  identity      the front-end top-1
  hand rule     protect (prompt hotwords present in top-1) + max(exact, ctc/len)
  learned       argmax P(min-CER) over the pool            (unconstrained)
  learned+cons  same, restricted to the hand rule's feasible set
  pool oracle   min CER over the pool
plus the designated-hotword recall for each.

Usage: --evidence ... --ctc-scores ... --uttid ... --aligned ... [--folds 5]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

NUM_FIELDS = ["asr_score", "search_score", "total_score", "exact_score",
              "exact_weighted_score", "hotword_score", "phonetic_score",
              "consensus_score", "consensus_support"]


def num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ctc-scores", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = defaultdict(list)
    for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig[f[1].strip()].append(norm(f[0]))

    ctc = {}
    for l in Path(a.ctc_scores).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
        d = {}
        for c in cs:
            if not c.get("text"):
                continue
            tok = c.get("token_ids") or []
            n = len(tok) if tok else max(1, len(norm(c["text"])))
            raw = num(c.get("asr_score"), -1e9)
            d[norm(c["text"])] = (raw, raw / n if n else raw, n)
        ctc[str(r["id"])] = d

    utts = []          # per-utterance record
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        idx = int(r["id"])
        uid = pos2utt[idx] if idx < len(pos2utt) else None
        inp = r.get("input") or {}
        old = norm(inp.get("asr_top1") or "")
        ref = norm(r.get("reference") or "")
        scores = ctc.get(uid) if uid else None
        if not old or not ref or not uid or not scores:
            continue
        cands = []
        for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
            if not isinstance(c, dict) or not c.get("text"):
                continue
            t = norm(c["text"])
            if t not in scores:
                continue
            cands.append((t, c))
        if not cands:
            continue
        prot = [h for h in texts_of(inp.get("prompt_hotwords")) if h in old]
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        utts.append(dict(uid=uid, old=old, ref=ref, cands=cands, prot=prot, hw=hw,
                         keys=desig.get(uid, []), inp=inp))

    # ---------------- feature construction -----------------------------------
    feats, names = [], None
    for u in utts:
        old, ref = u["old"], u["ref"]
        cands = u["cands"]
        ctc_ln = [ctc[u["uid"]][t][1] for t, _ in cands]
        ctc_rw = [ctc[u["uid"]][t][0] for t, _ in cands]
        ntok = [ctc[u["uid"]][t][2] for t, _ in cands]
        asr = [num(c.get("asr_score")) for _, c in cands]
        r_ctc, r_asr = ranks(ctc_ln), ranks(asr)
        best_ctc, best_asr = max(ctc_ln), max(asr)
        rows, ys, cers = [], [], []
        for i, (t, c) in enumerate(cands):
            row = [
                len(t),
                len(t) - len(old),
                ntok[i],
                asr[i],
                ctc_rw[i],
                ctc_ln[i],
                r_ctc[i],
                best_ctc - ctc_ln[i],
                r_asr[i],
                best_asr - asr[i],
                edit_distance(list(old), list(t)),
                num(c.get("search_score")), num(c.get("total_score")),
                num(c.get("exact_score")), num(c.get("exact_weighted_score")),
                num(c.get("hotword_score")), num(c.get("phonetic_score")),
                num(c.get("consensus_score")), num(c.get("consensus_support")),
                float(t == old),
                float(len(t) >= len(old)),
                float(all(p in t for p in u["prot"])),
                float(sum(1 for h in u["hw"] if h in t)),
                float(sum(1 for h in u["hw"] if h in t and h not in ref)),
            ]
            rows.append(row)
            cers.append(edit_distance(list(ref), list(t)))
        names = names or ["len", "dlen", "ntok", "asr", "ctc_raw", "ctc_ln", "ctc_rank",
                          "ctc_margin", "asr_rank", "asr_margin", "ed_old",
                          "search", "total", "exact_s", "exact_w", "hotword", "phon",
                          "consensus", "consensus_sup", "is_old", "nonshort",
                          "prot_ok", "hw_hits", "hw_spurious"]
        u["X"] = np.asarray(rows, dtype=np.float64)
        u["cers"] = cers
        mn = min(cers)
        u["y"] = np.asarray([1 if v == mn else 0 for v in cers], dtype=np.int64)

    uids = [u["uid"] for u in utts]
    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(utts))
    folds = [[] for _ in range(a.folds)]
    for k, i in enumerate(order):
        folds[k % a.folds].append(i)

    acc = defaultdict(lambda: dict(chars=0, e=0, men=0, hit=0))
    n_pool = n_rows = 0
    for f in range(a.folds):
        te = folds[f]
        tr = [i for g in range(a.folds) if g != f for i in folds[g]]
        Xtr = np.vstack([utts[i]["X"] for i in tr])
        ytr = np.concatenate([utts[i]["y"] for i in tr])
        # quantile-transform not needed for HistGB; keep it shallow to limit overfit
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0, random_state=a.seed)
        clf.fit(Xtr, ytr)
        for i in te:
            u = utts[i]
            p = clf.predict_proba(u["X"])[:, 1]
            charsn = len(u["ref"])
            n_pool += 1
            n_rows += len(u["cers"])
            choices = {"identity": u["old"]}
            # hand rule: restrict to feasible set (protect + no-shorten), then max(exact, ctc_ln)
            feas = [j for j in range(len(u["cers"]))
                    if all(p_ in u["cands"][j][0] for p_ in u["prot"])
                    and len(u["cands"][j][0]) >= len(u["old"])]
            if not feas:
                feas = [j for j in range(len(u["cers"]))
                        if all(p_ in u["cands"][j][0] for p_ in u["prot"])] or list(range(len(u["cers"])))
            j_hand = max(feas, key=lambda j: (num(u["cands"][j][1].get("exact_weighted_score")),
                                              ctc[u["uid"]][u["cands"][j][0]][1]))
            choices["hand"] = u["cands"][j_hand][0]
            choices["learned"] = u["cands"][int(np.argmax(p))][0]
            feas_all = feas
            choices["learned_cons"] = u["cands"][feas_all[int(np.argmax(p[feas_all]))]][0]
            choices["oracle"] = u["cands"][int(np.argmin(u["cers"]))][0]
            for k, txt in choices.items():
                d = acc[k]
                d["chars"] += charsn
                d["e"] += edit_distance(list(u["ref"]), list(txt))
                for key in u["keys"]:
                    d["men"] += 1
                    d["hit"] += int(key in txt)

    print("# %s   utterances=%d candidate-rows=%d folds=%d features=%d" % (
        a.label, n_pool, n_rows, a.folds, len(names)))
    print("%-16s %9s %9s %9s" % ("policy", "CER%", "recall%", "vs hand pp"))
    ref_cer = None
    for k in ("identity", "hand", "learned", "learned_cons", "oracle"):
        if k not in acc:
            continue
        d = acc[k]
        cer = 100 * d["e"] / d["chars"]
        rec = 100 * d["hit"] / d["men"]
        if k == "hand":
            ref_cer = cer
        print("%-16s %9.4f %9.2f %9s" % (k, cer, rec,
              "-" if ref_cer is None else "%+.4f" % (cer - ref_cer)))
    print("\nfeature list:", ", ".join(names))


if __name__ == "__main__":
    main()
