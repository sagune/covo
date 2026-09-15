#!/usr/bin/env python3
"""Per-character acoustic repair: gate on the model's own confidence, propose its own
alternative, and measure the ACTUAL effect on CER.

Results 16 found the per-character acoustic log-prob detects wrong characters far better than
anything before (AUC 0.940 vs 0.851), but the CER still got worse -- because "the position was
already wrong" is NOT the same as "the substitution fixes it": the old alternative came from
other candidates' text and was often itself wrong.

Two things change here:
  * the PROPOSAL is the model's own second choice at the frame where the character was emitted
    (`tok_alt`, exported by score_sensevoice_tokstats.py), not some other candidate's character
  * the METRIC is the realised change in errors, not the "position was wrong" proxy

Gates evaluated: the character's log-probability, and the margin between the emitted character
and the model's alternative at that frame.  Serving rule = substitute when the gate fires.

    acoustic_repair.py --tokstats <file>
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DEF = R / "stcmds_va_tokstats.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokstats", default=str(DEF))
    a = ap.parse_args()

    rows = []
    for line in Path(a.tokstats).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cs = (((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or [])
        if not cs:
            continue
        c0 = cs[0]
        top = norm(c0.get("norm_text") or c0.get("text") or "")
        ref = norm(r.get("reference") or "")
        lp = c0.get("tok_logps")
        al = c0.get("tok_alt")
        mg = c0.get("tok_alt_margin")
        if not top or not ref or not lp or not al or not mg:
            continue
        if not (len(lp) == len(al) == len(mg) == len(top)):
            continue
        rows.append(dict(top=top, ref=ref, lp=lp, al=al, mg=mg,
                         e0=edit_distance(list(ref), list(top)), n=len(top)))

    TOTAL_CH = sum(len(r["ref"]) for r in rows)
    BASE_ERR = sum(r["e0"] for r in rows)
    print("rows %d | characters %d | front-end CER %.4f%%   target 4.5000%%"
          % (len(rows), TOTAL_CH, 100.0 * BASE_ERR / TOTAL_CH))
    usable = sum(1 for r in rows for i in range(r["n"]) if r["al"][i])
    print("positions with a usable acoustic alternative: %d (%.1f%% of characters)"
          % (usable, 100.0 * usable / TOTAL_CH))

    def run(gate_name, thr):
        err = changed = fixed = broke = same = 0
        for r in rows:
            out = list(r["top"])
            for i in range(r["n"]):
                v = r["lp"][i] if gate_name == "logp" else r["mg"][i]
                if v >= thr or not r["al"][i]:
                    continue
                before = edit_distance(list(r["ref"]), out)
                out[i] = r["al"][i][0] if len(r["al"][i]) == 1 else out[i]
                after = edit_distance(list(r["ref"]), out)
                changed += 1
                if after < before:
                    fixed += 1
                elif after > before:
                    broke += 1
                else:
                    same += 1
            err += edit_distance(list(r["ref"]), out)
        cer = 100.0 * err / TOTAL_CH
        print("  %-5s %-7.2f %-8d %-7d %-7d %-7d %8.4f%% %s"
              % (gate_name, thr, changed, fixed, broke, same, cer,
                 "<= TARGET" if cer <= 4.5 else ("BETTER" if cer < 100.0 * BASE_ERR / TOTAL_CH else "")))
        return cer

    print()
    print("  %-5s %-7s %-8s %-7s %-7s %-7s %-9s" % ("gate", "thr", "changed", "fixed", "broke", "neutral", "corpus CER"))
    best = (None, None, 1e9)
    for thr in (-8.0, -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.6, -0.3):
        c = run("logp", thr)
        if c < best[2]:
            best = ("logp", thr, c)
    for thr in (12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.6, 0.3):
        c = run("margin", thr)
        if c < best[2]:
            best = ("margin", thr, c)
    print()
    print("best: gate=%s thr=%.2f -> CER %.4f%%   (front end %.4f%%)"
          % (best[0], best[1], best[2], 100.0 * BASE_ERR / TOTAL_CH))


if __name__ == "__main__":
    main()
