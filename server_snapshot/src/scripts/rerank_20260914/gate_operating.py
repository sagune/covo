#!/usr/bin/env python3
"""Can the gate + the trivial selection rule actually reach 4.5?

Results 12-14 established:
  * selection on decision rows: min-edit rule 43.5%, features 4.4%, trained generator 20.9%,
    zero-shot scorer 30.3%
  * gating ("is the top-1 wrong?"): the best signals are the acoustic margin and the scorer's
    confidence, AUC ~0.78 -- the first signals in this whole investigation that carry real
    information, but far from perfect
  * the frontier needs a >= 36.9% at k >= 0.95, i.e. the gate must catch ~85% of decision rows
    while touching only 5% of the good ones

This combines the two halves for real: sweep the gate threshold, apply the min-edit rule only
where the gate fires, and compute the resulting corpus CER from the actual rows -- no averages,
no oracle knowledge of which rows are decision rows.

    gate_operating.py [--records ...] [--scorer ...]
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
    ap.add_argument("--records", default=DEF)
    ap.add_argument("--scorer", default=DEF_SCORER)
    a = ap.parse_args()

    scorer = {}
    sp = Path(a.scorer)
    if sp.exists():
        for line in sp.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                scorer[norm((r.get("input") or {}).get("asr_top1") or "") + "|" +
                       norm(r.get("reference") or "")] = r.get("candidate_scores") or []

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
        cands, seen = [], set()
        for t in [top] + texts_of(inp.get("nbest")):
            if t and t not in seen:
                seen.add(t)
                cands.append(t)
        if len(cands) < 2:
            rows.append(dict(ch=len(ref), e0=edit_distance(list(ref), list(top)),
                             dec=False, gain=0.0, dmg=0.0, gate=-9e9))
            continue
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        e0 = eds[0]
        feat = {}
        for c in ((inp.get("cbwhisper") or {}).get("candidates") or []):
            k = norm(c.get("text") or "")
            if k and k not in feat:
                feat[k] = c
        asr = [float(feat[t]["asr_score"]) for t in cands
               if t in feat and feat[t].get("asr_score") is not None]
        margin = (max(asr) - min(asr)) if len(asr) > 1 else 0.0
        sc = scorer.get(top + "|" + ref)
        top1p = -1.0
        if sc:
            ls = [float(c.get("lm_score") or 0.0) for c in sc]
            mx = max(ls)
            ex = [math.exp(s - mx) for s in ls]
            z = sum(ex)
            top1p = math.exp(ls[0] - mx) / z if z else -1.0
        # gate score: LOWER means "more likely the top-1 is wrong"
        gate = -min(margin, 10.0) if not sc else -top1p
        if min(eds) < e0:
            bi = min(range(len(eds)), key=lambda i: (eds[i], i))
            j = min(range(1, len(cands)),
                    key=lambda i: (edit_distance(list(cands[0]), list(cands[i])), i))
            rows.append(dict(ch=len(ref), e0=e0, dec=True, gain=e0 - eds[bi],
                             dmg=max(0, eds[j] - e0), gate=gate))
        else:
            j = min(range(1, len(cands)),
                    key=lambda i: (edit_distance(list(cands[0]), list(cands[i])), i))
            rows.append(dict(ch=len(ref), e0=e0, dec=False, gain=0.0,
                             dmg=max(0, eds[j] - e0), gate=gate))

    TOTAL_CH = sum(r["ch"] for r in rows)
    BASE_ERR = sum(r["e0"] for r in rows)
    dec = [r for r in rows if r["dec"]]
    non = [r for r in rows if not r["dec"]]
    print("rows %d | decision %d | non-decision %d | front-end CER %.4f%%"
          % (len(rows), len(dec), len(non), 100.0 * BASE_ERR / TOTAL_CH))
    print("gate = the better of (acoustic margin, scorer top-1 probability); lower => edit")
    print()
    print("  %-9s %-8s %-8s %-8s %-11s %s" % ("thresh", "catch", "k", "a_eff", "corpus CER", "note"))
    best = (None, 1e9)
    for q in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
        # threshold such that (1-q) of the NON-decision rows are left untouched  -> k = 1-q
        vals = sorted(r["gate"] for r in non)
        if not vals:
            continue
        thr = vals[int(q * (len(vals) - 1))]
        act = [r for r in rows if r["gate"] <= thr]
        act_dec = [r for r in act if r["dec"]]
        act_non = [r for r in act if not r["dec"]]
        err = BASE_ERR
        for r in act_dec:
            err -= r["gain"] * 0.435          # the min-edit rule's measured accuracy
        for r in act_non:
            err += r["dmg"]
        cer = 100.0 * err / TOTAL_CH
        k = 100.0 * (1 - len(act_non) / len(non))
        catch = 100.0 * len(act_dec) / max(1, len(dec))
        aeff = catch * 0.435
        note = "<= TARGET" if cer <= TARGET else ""
        print("  %-9.3f %-8.1f %-8.1f %-8.1f %-11.4f %s" % (thr, catch, k, aeff, cer, note))
        if cer < best[1]:
            best = (thr, cer)
    print()
    print("best gate operating point: thresh %.3f -> CER %.4f%%   (front end %.4f%%, target %.2f%%)"
          % (best[0], best[1], 100.0 * BASE_ERR / TOTAL_CH, TARGET))
    print("frontier from RESULTS 12.6 needs a>=36.9%% at k=0.95.")


if __name__ == "__main__":
    main()
