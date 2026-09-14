#!/usr/bin/env python3
"""Idea A pilot, round 2: listwise objectives that optimise the metric directly.

Round 1 trained a per-candidate binary classifier (label = "is the min-CER
candidate") with a shallow GBDT and lost to the hand rule by +0.052 pp.  That is
weak evidence: the objective was wrong (P(is-argmin) is not what we select on), the
model was shallow, and there were only 1334 utterances.

This version keeps the identical protocol (grouped 5-fold CV by utterance, same
pool, same rows) and swaps in objectives that target CER:

  gbdt_cls      HistGB, binary "is min-CER"            (round-1 baseline)
  lin_ce        linear, listwise cross-entropy toward the min-CER candidate
  lin_ece       linear, expected CER:  sum_j softmax(s)_j * CER_j
  mlp_ce        one hidden layer, listwise CE
All are trained on 4 folds and scored on the held-out fold; the reported CER/recall
is over the held-out folds only.  `--with-hand-flag` adds a binary feature marking
the hand rule's own pick, as a machinery sanity check (it should then be >= hand).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


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


def keep_indices(names, drop):
    """Indices of the features to KEEP after removing every name matching a substring."""
    if not drop:
        return None
    pats = [d for d in drop.split(",") if d]
    return [i for i, n in enumerate(names) if not any(p in n for p in pats)]


def prune(utts, keep):
    if keep is None:
        return
    for u in utts:
        u["X"] = u["X"][:, keep]


def build(evidence, ctc_path, uttid, aligned):
    pos2utt = [l.split()[0] for l in Path(uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = defaultdict(list)
    for line in Path(aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig[f[1].strip()].append(norm(f[0]))
    ctc = {}
    for l in Path(ctc_path).read_text(encoding="utf-8").splitlines():
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

    utts = []
    for l in Path(evidence).read_text(encoding="utf-8").splitlines():
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
        cands = [(norm(c["text"]), c) for c in ((inp.get("cbwhisper") or {}).get("candidates") or [])
                 if isinstance(c, dict) and c.get("text") and norm(c["text"]) in scores]
        if not cands:
            continue
        prot = [h for h in texts_of(inp.get("prompt_hotwords")) if h in old]
        hw = sorted(set(texts_of(inp.get("hotwords"))) | set(texts_of(inp.get("prompt_hotwords"))))
        utts.append(dict(uid=uid, old=old, ref=ref, cands=cands, prot=prot, hw=hw,
                         keys=desig.get(uid, []), raw=r))
    return utts, ctc


def featurise(utts, ctc, with_hand_flag, label_mode="mincer"):
    names = ["len", "dlen", "ntok", "asr", "ctc_raw", "ctc_ln", "ctc_rank", "ctc_margin",
             "asr_rank", "asr_margin", "ed_old", "search", "total", "exact_s", "exact_w",
             "hotword", "phon", "consensus", "consensus_sup", "is_old", "nonshort",
             "prot_ok", "hw_hits", "hw_spurious"]
    if with_hand_flag:
        names = names + ["hand_pick"]
    for u in utts:
        old, ref, cs = u["old"], u["ref"], u["cands"]
        cl = [ctc[u["uid"]][t][1] for t, _ in cs]
        cr = [ctc[u["uid"]][t][0] for t, _ in cs]
        nt = [ctc[u["uid"]][t][2] for t, _ in cs]
        asr = [num(c.get("asr_score")) for _, c in cs]
        rc, ra = ranks(cl), ranks(asr)
        bc, ba = max(cl), max(asr)
        feas = [j for j in range(len(cs)) if all(p in cs[j][0] for p in u["prot"])
                and len(cs[j][0]) >= len(old)]
        feas_prot = [j for j in range(len(cs)) if all(p in cs[j][0] for p in u["prot"])] or list(range(len(cs)))
        if not feas:
            feas = feas_prot
        jh = max(feas, key=lambda j: (num(cs[j][1].get("exact_weighted_score")), cl[j]))
        rows, cers, hlost = [], [], []
        for i, (t, c) in enumerate(cs):
            row = [len(t), len(t) - len(old), nt[i], asr[i], cr[i], cl[i], rc[i], bc - cl[i],
                   ra[i], ba - asr[i], edit_distance(list(old), list(t)),
                   num(c.get("search_score")), num(c.get("total_score")), num(c.get("exact_score")),
                   num(c.get("exact_weighted_score")), num(c.get("hotword_score")),
                   num(c.get("phonetic_score")), num(c.get("consensus_score")),
                   num(c.get("consensus_support")), float(t == old), float(len(t) >= len(old)),
                   float(all(p in t for p in u["prot"])),
                   float(sum(1 for h in u["hw"] if h in t)),
                   float(sum(1 for h in u["hw"] if h in t and h not in ref))]
            if with_hand_flag:
                row.append(float(i == jh))
            rows.append(row)
            cers.append(edit_distance(list(ref), list(t)))
            # designated hotwords that the reference has but this candidate lacks
            hlost.append(sum(1 for k in u["keys"] if k in ref and k not in t))
        u["X"] = np.asarray(rows, dtype=np.float64)
        u["cers"] = np.asarray(cers, dtype=np.float64)
        mn = min(cers)
        if label_mode == "nohotloss":
            # prefer the minimum-CER candidate among those that lose no designated
            # hotword the reference contains; fall back to plain min-CER if none does
            keep_h = [j for j in range(len(cers)) if hlost[j] == 0]
            if keep_h:
                mn = min(cers[j] for j in keep_h)
                u["y_cls"] = np.asarray([1 if (j in keep_h and cers[j] == mn) else 0
                                         for j in range(len(cers))], dtype=np.int64)
            else:
                u["y_cls"] = np.asarray([1 if v == mn else 0 for v in cers], dtype=np.int64)
        else:
            u["y_cls"] = np.asarray([1 if v == mn else 0 for v in cers], dtype=np.int64)
        u["y"] = u["y_cls"].astype(np.float64) / max(1.0, u["y_cls"].sum())
        u["conflict"] = 1.0 if hlost[int(np.argmin(cers))] > 0 else 0.0
        u["yl"] = u["y_cls"].astype(np.float64)
        u["ncer"] = u["cers"] / max(1.0, len(ref))
        hl = np.asarray(hlost, dtype=np.float64)
        u["hlost"] = hl / max(1.0, hl.max())
        u["feas"] = feas
        u["feas_prot"] = feas_prot
    return names


def pad_pools(utts, idx, key, cw=0.0):
    """(N, K, F) padded features + (N, K) targets + mask, built once per fold."""
    K = max(len(utts[i][key]) for i in idx)
    N, F = len(idx), utts[idx[0]][key].shape[1]
    X = np.zeros((N, K, F), dtype=np.float32)
    Y = np.zeros((N, K), dtype=np.float32)
    C = np.zeros((N, K), dtype=np.float32)
    H = np.zeros((N, K), dtype=np.float32)
    M = np.zeros((N, K), dtype=np.float32)
    Wt = np.ones((N,), dtype=np.float32)
    for r, i in enumerate(idx):
        u = utts[i]
        k = len(u[key])
        X[r, :k] = u[key]
        Y[r, :k] = u["y"]
        C[r, :k] = u["ncer"]
        H[r, :k] = u["hlost"]
        M[r, :k] = 1.0
        Wt[r] = 1.0 + cw * u.get("conflict", 0.0)
    return (torch.tensor(X), torch.tensor(Y), torch.tensor(C), torch.tensor(H),
            torch.tensor(M), torch.tensor(Wt))


def train_torch(model, blob, loss_kind, epochs=250, lr=0.08, wd=1e-4, seed=0, recall_weight=0.0):
    torch.manual_seed(seed)
    X, Y, C, H, M, Wt = blob
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    neg = torch.finfo(X.dtype).min
    for ep in range(epochs):
        s = model(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1])
        s = s.masked_fill(M < 0.5, neg)
        logp = torch.log_softmax(s, dim=1)
        if loss_kind == "ce":
            per = -(Y * logp).sum(dim=1)
        else:
            per = (torch.softmax(s, dim=1) * C).sum(dim=1)
        if recall_weight > 0:
            per = per + recall_weight * (torch.softmax(s, dim=1) * H).sum(dim=1)
        loss = (per * Wt).sum() / Wt.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def score_pools(model, utts, idx, key):
    K = max(len(utts[i][key]) for i in idx)
    X = np.zeros((len(idx), K, utts[idx[0]][key].shape[1]), dtype=np.float32)
    for r, i in enumerate(idx):
        X[r, :len(utts[i][key])] = utts[i][key]
    with torch.no_grad():
        s = model(torch.tensor(X).reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1]).numpy()
    out = {}
    for r, i in enumerate(idx):
        out[i] = s[r, :len(utts[i][key])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ctc-scores", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--with-hand-flag", action="store_true")
    ap.add_argument("--models", default="gbdt_cls,lin_ce,lin_ece,mlp_ce")
    ap.add_argument("--seeds", default="", help="comma list; if set, repeats CV per seed")
    ap.add_argument("--train-evidence", default="")
    ap.add_argument("--train-ctc", default="")
    ap.add_argument("--train-uttid", default="")
    ap.add_argument("--train-aligned", default="")
    ap.add_argument("--emit-jsonl", default="", help="write reranked evidence here")
    ap.add_argument("--drop-features", default="", help="comma list of name substrings to drop")
    ap.add_argument("--conflict-weight", type=float, default=0.0,
                    help="extra weight on utterances whose pool CER-optimum loses a designated hotword")
    ap.add_argument("--label-mode", default="mincer", choices=["mincer", "nohotloss"],
                    help="mincer: label = pool minimum-CER candidate. "
                         "nohotloss: minimum-CER candidate among those losing no designated hotword")
    ap.add_argument("--recall-weight", type=float, default=0.0,
                    help="weight of the auxiliary expected designated-hotword-loss term")
    ap.add_argument("--protect-only", action="store_true",
                    help="feasible set = protection only, i.e. deletion is allowed")
    ap.add_argument("--constrain", action="store_true",
                    help="restrict both the training label and the inference choice to the "
                         "feasible set (keeps every protected hotword, never shortens)")
    a = ap.parse_args()

    utts, ctc = build(a.evidence, a.ctc_scores, a.uttid, a.aligned)
    all_names = featurise(utts, ctc, a.with_hand_flag)
    keep = keep_indices(all_names, a.drop_features)
    prune(utts, keep)
    names = [n for i, n in enumerate(all_names) if keep is None or i in keep]
    ndim = len(names)

    if a.constrain:
        for u in utts:
            m = np.zeros(len(u["y_cls"]))
            for j in (u["feas_prot"] if a.protect_only else u["feas"]):
                m[j] = 1.0
            u["y_cls"] = (u["y_cls"] * m).astype(np.int64)
            if u["y_cls"].sum() == 0:            # no feasible candidate is optimal: fall back
                u["y_cls"] = np.asarray([1 if j == u["feas"][0] else 0 for j in range(len(m))], dtype=np.int64)
            u["y"] = u["y_cls"].astype(np.float64) / max(1.0, u["y_cls"].sum())

    # ---- optional cross-domain mode: train on one dataset, score this one -------
    if a.train_evidence:
        tr_utts, tr_ctc = build(a.train_evidence, a.train_ctc, a.train_uttid, a.train_aligned)
        featurise(tr_utts, tr_ctc, a.with_hand_flag)
        prune(tr_utts, keep)
        mu = np.vstack([u["X"] for u in tr_utts]).mean(0)
        sd = np.vstack([u["X"] for u in tr_utts]).std(0) + 1e-6
        for u in tr_utts:
            u["Z"] = (u["X"] - mu) / sd
        for u in utts:
            u["Z"] = (u["X"] - mu) / sd
        acc = defaultdict(lambda: dict(chars=0, e=0, men=0, hit=0))

        def emit2(k, u, txt):
            d = acc[k]
            d["chars"] += len(u["ref"])
            d["e"] += edit_distance(list(u["ref"]), list(txt))
            for key in u["keys"]:
                d["men"] += 1
                d["hit"] += int(key in txt)

        tridx = list(range(len(tr_utts)))
        teidx = list(range(len(utts)))
        for i in teidx:
            u = utts[i]
            emit2("identity", u, u["old"])
            jh = max(u["feas"], key=lambda j: (num(u["cands"][j][1].get("exact_weighted_score")),
                                               ctc[u["uid"]][u["cands"][j][0]][1]))
            emit2("hand", u, u["cands"][jh][0])
            emit2("oracle", u, u["cands"][int(np.argmin(u["cers"]))][0])
        wanted2 = [m for m in a.models.split(",") if m]
        for tag, kind, hidden in (("lin_ce", "ce", 0), ("lin_ece", "ece", 0), ("mlp_ce", "ce", 32)):
            if tag not in wanted2:
                continue
            blob = pad_pools(tr_utts, tridx, "Z", a.conflict_weight)
            layers = [nn.Linear(ndim, hidden), nn.ReLU(), nn.Linear(hidden, 1)] if hidden else [nn.Linear(ndim, 1)]
            model = nn.Sequential(*layers)
            train_torch(model, blob, kind, seed=a.seed, recall_weight=a.recall_weight)
            sc = score_pools(model, utts, teidx, "Z")
            for i in teidx:
                s = sc[i]
                if a.constrain:
                    _f = utts[i]["feas_prot"] if a.protect_only else utts[i]["feas"]
                    j = _f[int(np.argmax(s[_f]))]
                else:
                    j = int(np.argmax(s))
                emit2(tag, utts[i], utts[i]["cands"][j][0])
        # ---- optional: write the reranked pool out in the rescorer's own format ----
        emit_rows = {}
        if a.emit_jsonl:
            tag = "lin_ce"
            blob = pad_pools(tr_utts, tridx, "Z")
            model = nn.Sequential(nn.Linear(ndim, 1))
            train_torch(model, blob, "ce", seed=a.seed, recall_weight=a.recall_weight)
            sc = score_pools(model, utts, teidx, "Z")
            src = {}
            for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    src[str(r.get("id"))] = r
            for i in teidx:
                u = utts[i]
                best = u["cands"][int(np.argmax(sc[i]))][0]
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))
                old_nb = (r.get("input") or {}).get("nbest") or []
                r["input"]["asr_top1"] = best
                r["input"]["nbest"] = [best] + [t for t in old_nb if norm(t) != best]
                emit_rows[str(r.get("id"))] = r
            order = [str(json.loads(l).get("id"))
                     for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
            with open(a.emit_jsonl, "w", encoding="utf-8") as fh:
                for rid in order:
                    fh.write(json.dumps(emit_rows[rid], ensure_ascii=False) + "\n")
            print("# wrote %d rows to %s" % (len(order), a.emit_jsonl))

        print("# TRANSFER  train=%s (%d utts)  ->  test=%s (%d utts)  features=%d" % (
            Path(a.train_evidence).name, len(tr_utts), Path(a.evidence).name, len(utts), ndim))
        print("%-14s %9s %9s %13s" % ("policy", "CER%", "recall%", "vs hand pp"))
        hand = None
        for k in ["identity", "hand"] + wanted2 + ["oracle"]:
            if k not in acc:
                continue
            d = acc[k]
            cer = 100 * d["e"] / d["chars"]
            rec = 100 * d["hit"] / d["men"]
            if k == "hand":
                hand = cer
            print("%-14s %9.4f %9.2f %13s" % (k, cer, rec, "-" if hand is None else "%+.4f" % (cer - hand)))
        return

    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(utts))
    folds = [[] for _ in range(a.folds)]
    for k, i in enumerate(order):
        folds[k % a.folds].append(i)

    wanted = [m for m in a.models.split(",") if m]
    acc = defaultdict(lambda: dict(chars=0, e=0, men=0, hit=0))
    oof_pick = {}          # utterance index -> text chosen by the out-of-fold model

    def emit(k, u, txt):
        d = acc[k]
        d["chars"] += len(u["ref"])
        d["e"] += edit_distance(list(u["ref"]), list(txt))
        for key in u["keys"]:
            d["men"] += 1
            d["hit"] += int(key in txt)

    for f in range(a.folds):
        te = folds[f]
        tr = [i for g in range(a.folds) if g != f for i in folds[g]]
        for i in te:
            u = utts[i]
            emit("identity", u, u["old"])
            jh = max(u["feas"], key=lambda j: (num(u["cands"][j][1].get("exact_weighted_score")),
                                               ctc[u["uid"]][u["cands"][j][0]][1]))
            emit("hand", u, u["cands"][jh][0])
            emit("oracle", u, u["cands"][int(np.argmin(u["cers"]))][0])

        if "gbdt_cls" in wanted:
            Xtr = np.vstack([utts[i]["X"] for i in tr])
            ytr = np.concatenate([utts[i]["y_cls"] for i in tr])
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                                 min_samples_leaf=20, l2_regularization=1.0,
                                                 random_state=a.seed).fit(Xtr, ytr)
            for i in te:
                p = clf.predict_proba(utts[i]["X"])[:, 1]
                emit("gbdt_cls", utts[i], utts[i]["cands"][int(np.argmax(p))][0])

        mu = np.vstack([utts[i]["X"] for i in tr]).mean(0)
        sd = np.vstack([utts[i]["X"] for i in tr]).std(0) + 1e-6
        for i in range(len(utts)):
            utts[i]["Z"] = ((utts[i]["X"] - mu) / sd)
        for tag, kind, hidden in (("lin_ce", "ce", 0), ("lin_ece", "ece", 0), ("mlp_ce", "ce", 32)):
            if tag not in wanted:
                continue
            blob = pad_pools(utts, tr, "Z", a.conflict_weight)
            layers = [nn.Linear(ndim, hidden), nn.ReLU(), nn.Linear(hidden, 1)] if hidden else [nn.Linear(ndim, 1)]
            model = nn.Sequential(*layers)
            train_torch(model, blob, kind, seed=a.seed, recall_weight=a.recall_weight)
            sc = score_pools(model, utts, te, "Z")
            for i in te:
                s = sc[i]
                if a.constrain:
                    _f = utts[i]["feas_prot"] if a.protect_only else utts[i]["feas"]
                    j = _f[int(np.argmax(s[_f]))]
                else:
                    j = int(np.argmax(s))
                pick = utts[i]["cands"][j][0]
                emit(tag, utts[i], pick)
                if tag == "lin_ce":
                    oof_pick[i] = pick

    # ---- optional: write the out-of-fold reranked pool for the end-to-end arm ----
    if a.emit_jsonl and oof_pick:
        order = [str(json.loads(l).get("id"))
                 for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(a.emit_jsonl, "w", encoding="utf-8") as fh:
            for i, u in enumerate(utts):
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))
                best = oof_pick[i]
                old_nb = (r.get("input") or {}).get("nbest") or []
                r["input"]["asr_top1"] = best
                r["input"]["nbest"] = [best] + [t for t in old_nb if norm(t) != best]
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("# wrote %d out-of-fold rows to %s" % (len(utts), a.emit_jsonl))

    print("# %s   utterances=%d  features=%d  hand-flag=%s" % (
        a.label, len(utts), ndim, a.with_hand_flag))
    print("%-14s %9s %9s %13s" % ("policy", "CER%", "recall%", "vs hand pp"))
    hand = None
    for k in ["identity", "hand"] + wanted + ["oracle"]:
        if k not in acc:
            continue
        d = acc[k]
        cer = 100 * d["e"] / d["chars"]
        rec = 100 * d["hit"] / d["men"]
        if k == "hand":
            hand = cer
        print("%-14s %9.4f %9.2f %13s" % (k, cer, rec, "-" if hand is None else "%+.4f" % (cer - hand)))


if __name__ == "__main__":
    main()
