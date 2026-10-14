#!/usr/bin/env python3
"""MBR / minimum-Bayes-risk selection over the likelihood scorer's own candidate scores.

This is the literature-grounded fix for the failure that killed the SFT run.  The SFT model
learned a CORPUS-LEVEL PRIOR ("how often should I edit") and that prior is what breaks when
the target dataset changes (75.4% training prior vs 35.7% on ST-CMDS -> it edited 41.6% of
rows at 27.1% precision).  MBR replaces the prior with a PER-ROW decision:

    p(c) = softmax(lm_score(c) / T)                  the model's own belief over candidates
    R(x) = sum_c p(c) * edit_distance(x, c)          expected risk (loss = edit distance)
    choose argmin_x R(x)

Properties that matter for transfer:
  * no edit-rate parameter and no threshold tuned on the target -> nothing to re-adapt
  * output is always a candidate -> cannot emit off-pool text (the SFT model put 15.3% of its
    outputs outside the pool, which is pure loss)
  * when the model is confident in the top-1, p is peaked and R is minimised at the top-1, so
    it abstains by construction rather than by prior

Reports the two numbers the operating curve needs -- a (accuracy on decision rows) and
k (keep rate on rows that should not be touched) -- plus the corpus CER, for a temperature
sweep, so the result can be read straight against the iso-target frontier.

    mbr_select.py --records <scorer output jsonl> [--top1-file <front-end predictions>]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

TARGET = 4.5


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
    ap.add_argument("--records", required=True, help="the likelihood scorer's output")
    ap.add_argument("--temps", default="0.05,0.1,0.2,0.5,1,2,5")
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
        cs = r.get("candidate_scores") or []
        cands, scores = [], []
        for c in cs:
            t = norm(c.get("text") or "")
            if t and t not in cands:
                cands.append(t)
                scores.append(float(c.get("lm_score") or 0.0))
        if top not in cands:
            cands.insert(0, top)
            scores.insert(0, max(scores) + 1.0 if scores else 0.0)
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        rows.append(dict(ref=ref, cands=cands, scores=scores, eds=eds, e0=eds[0]))

    TOTAL_CH = sum(len(r["ref"]) for r in rows)
    BASE_ERR = sum(r["e0"] for r in rows)
    dec = [r for r in rows if min(r["eds"]) < r["e0"]]
    non = [r for r in rows if min(r["eds"]) >= r["e0"]]
    oracle_gain = sum(r["e0"] - min(r["eds"]) for r in dec)
    print("rows %d chars %d | front-end CER %.4f%%" % (len(rows), TOTAL_CH, 100.0 * BASE_ERR / TOTAL_CH))
    print("decision rows %d | non-decision rows %d" % (len(dec), len(non)))
    print("target %.2f%% needs %.4f pp = %.1f%% of the oracle gain"
          % (TARGET, 100.0 * BASE_ERR / TOTAL_CH - TARGET,
             100 * (100.0 * BASE_ERR / TOTAL_CH - TARGET) / (100.0 * oracle_gain / TOTAL_CH)))
    print()
    print("  %-7s %-9s %-9s %-11s %s" % ("T", "a(dec)", "k(non)", "corpus CER", "verdict"))
    best = None
    for T in [float(x) for x in a.temps.split(",")]:
        err = 0
        hit = keep = 0
        for r in rows:
            mx = max(r["scores"])
            ex = [math.exp((s - mx) / T) for s in r["scores"]]
            z = sum(ex)
            p = [e / z for e in ex]
            best_i, best_r = 0, None
            for i in range(len(r["cands"])):
                R = sum(p[j] * edit_distance(list(r["cands"][i]), list(r["cands"][j]))
                        for j in range(len(r["cands"])))
                if best_r is None or R < best_r:
                    best_r, best_i = R, i
            err += r["eds"][best_i]
            if min(r["eds"]) < r["e0"]:
                hit += best_i == min(range(len(r["eds"])), key=lambda i: (r["eds"][i], i))
            else:
                keep += best_i == 0
        cer = 100.0 * err / TOTAL_CH
        aa = 100.0 * hit / max(1, len(dec))
        kk = 100.0 * keep / max(1, len(non))
        v = "<= TARGET" if cer <= TARGET else ""
        print("  %-7.2f %-9.1f %-9.1f %-11.4f %s" % (T, aa, kk, cer, v))
        if best is None or cer < best[1]:
            best = (T, cer, aa, kk)
    print()
    print("best: T=%.2f  CER %.4f%%  a=%.1f%%  k=%.1f%%" % best)
    print("iso-target frontier (RESULTS 12.6): a>=26.2%% at k=1.0, a>=36.9%% at k=0.95, a>=47.6%% at k=0.90")


if __name__ == "__main__":
    main()
