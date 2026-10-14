#!/usr/bin/env python3
"""What does the BACKEND SELECTOR have to achieve?  The operating curve.

RESULTS 12 showed the two halves of the problem are separable:
  * SELECTION on the 1461 decision rows -- a zero-training rule ("pick the non-top-1
    candidate with the smallest edit distance to top-1") gets 43.5%, the best front-end
    feature gets 4.4%, and the trained generator got 20.9%.
  * ABSTENTION on the other 3669 rows -- nothing tried so far can do it.

This turns that into the bar a selector must clear.  A selector over the candidate list
INCLUDING the top-1 has exactly two failure modes, and both are measurable here:
    a = P(pick the best candidate | a better candidate exists)    -- the decision rows
    k = P(keep the top-1 | no candidate is better)                -- the non-decision rows
Writing the corpus CER as a function of (a, k) gives the iso-4.5 frontier, so we can say
concretely: "at the 43.5% accuracy a trivial rule already reaches, how well must it abstain?"

    selector_operating_curve.py [--records ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

DEF = "/root/autodl-tmp/.dsh_checks/rerank/e2eSTCMDS_VA.messages.jsonl"
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
    a = ap.parse_args()

    TOTAL_CH = BASE_ERR = 0
    dec = []      # decision rows: (gain_if_best, 0)
    non = []      # non-decision rows: (damage_if_min_edit_rule)
    for line in Path(a.records).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        if not ref or not top:
            continue
        TOTAL_CH += len(ref)
        cands, seen = [], set()
        for t in [top] + texts_of(inp.get("nbest")):
            if t and t not in seen:
                seen.add(t)
                cands.append(t)
        eds = [edit_distance(list(ref), list(t)) for t in cands]
        BASE_ERR += eds[0]
        if len(cands) < 2:
            continue
        if min(eds) < eds[0]:
            dec.append(max(0, eds[0] - min(eds)))
        else:
            # what the min-edit rule would cost on a row that should be left alone
            j = min(range(1, len(cands)),
                    key=lambda i: (edit_distance(list(cands[0]), list(cands[i])), i))
            non.append(max(0, eds[j] - eds[0]))

    base_cer = 100.0 * BASE_ERR / TOTAL_CH
    oracle_gain = sum(dec)
    print("decision rows %d | non-decision rows %d | chars %d" % (len(dec), len(non), TOTAL_CH))
    print("front-end CER %.4f%%   oracle gain %.4f pp   target %.2f%% needs %.4f pp"
          % (base_cer, 100.0 * oracle_gain / TOTAL_CH, TARGET, base_cer - TARGET))
    need = (base_cer - TARGET) / (100.0 * oracle_gain / TOTAL_CH)
    print("=> a selector must capture at least %.1f%% of the oracle gain (before any damage)" % (100 * need))
    print()
    print("mean gain when a decision row is fixed : %.3f chars" % (oracle_gain / max(1, len(dec))))
    print("mean damage when a good row is edited  : %.3f chars" % (sum(non) / max(1, len(non))))
    print()

    print("=" * 92)
    print("ISO-TARGET FRONTIER:  CSTAR(a) = accuracy on decision rows needed, as a function of")
    print("                      the abstention rate k on the rows that should not be touched")
    print("=" * 92)
    print("  %-8s %s" % ("k", "  ".join("a>=%4.0f%%" % (100 * x / 10) for x in range(1, 10))))
    for k in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
        damage = (1 - k) * sum(non)
        # base_cer - a*oracle_gain + damage <= TARGET
        need_gain = (100.0 * (BASE_ERR + damage) / TOTAL_CH) - TARGET
        a_req = 100.0 * TOTAL_CH * need_gain / 100.0 / max(1e-9, oracle_gain)
        cells = []
        for x in range(1, 10):
            mark = " OK " if (100 * x / 10) >= a_req else "  . "
            cells.append(mark)
        print("  %-8.2f %s   (needs a >= %5.1f%%)" % (k, " ".join(cells), a_req))

    print()
    print("=" * 92)
    print("REFERENCE POINTS measured in RESULTS 12")
    print("=" * 92)
    for label, acc, keep in (("zero-training min-edit rule, applied only where needed", 0.435, 1.0),
                             ("trained generator (SFT)", 0.209, 0.0),
                             ("best front-end feature (total_score)", 0.044, 0.0)):
        damage = (1 - keep) * sum(non)
        cer = 100.0 * (BASE_ERR - acc * oracle_gain + damage) / TOTAL_CH
        print("  %-52s a=%5.1f%% k=%4.0f%%  ->  CER %7.4f%%  %s"
              % (label, 100 * acc, 100 * keep, cer, "<= TARGET" if cer <= TARGET else ""))


if __name__ == "__main__":
    main()
