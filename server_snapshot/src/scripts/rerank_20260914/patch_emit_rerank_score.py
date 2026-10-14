#!/usr/bin/env python3
"""Patch learned_ranker_pilot2.py so the emitted records carry the rescorer's score.

The bridge can then print `rerank=<score>` on the winner's n-best line, which gives
the model a positive signal for a candidate whose beam metadata looks weak.  Purely
additive: one extra field on the emitted rows.
"""
from pathlib import Path
import subprocess

p = Path("/root/autodl-tmp/.dsh_checks/learned_ranker_pilot2.py")
s = p.read_text(encoding="utf-8")

# transfer-mode emit
OLD_T = """            for i in teidx:
                u = utts[i]
                best = u["cands"][int(np.argmax(sc[i]))][0]
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))"""
NEW_T = """            for i in teidx:
                u = utts[i]
                _j = int(np.argmax(sc[i]))
                best = u["cands"][_j][0]
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))
                r["input"]["asr_top1_rerank_score"] = float(sc[i][_j])"""

# CV-mode (out-of-fold) emit
OLD_C = """            for i, u in enumerate(utts):
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))
                best = oof_pick[i]
                old_nb = (r.get("input") or {}).get("nbest") or []"""
NEW_C = """            for i, u in enumerate(utts):
                r = json.loads(json.dumps(u["raw"], ensure_ascii=False))
                best = oof_pick[i]
                for _j, _c in enumerate(u["cands"]):
                    if _c[0] == best:
                        r["input"]["asr_top1_rerank_score"] = float(oof_score[i][_j])
                        break
                old_nb = (r.get("input") or {}).get("nbest") or []"""

# record the out-of-fold scores
OLD_S = """                if tag == "lin_ce":
                    oof_pick[i] = pick"""
NEW_S = """                if tag == "lin_ce":
                    oof_pick[i] = pick
                    oof_score[i] = s"""
OLD_D = "    oof_pick = {}          # utterance index -> text chosen by the out-of-fold model"
NEW_D = ("    oof_pick = {}          # utterance index -> text chosen by the out-of-fold model\n"
         "    oof_score = {}         # utterance index -> score vector, for the rerank= field")

for old, new in ((OLD_D, NEW_D), (OLD_S, NEW_S), (OLD_T, NEW_T), (OLD_C, NEW_C)):
    if old not in s:
        raise SystemExit("PATCH TARGET NOT FOUND:\n" + old[:160])
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
r = subprocess.run(["/root/autodl-tmp/great/bin/python", "-m", "py_compile", str(p)],
                   capture_output=True, text=True)
print("compiles:", r.returncode == 0, r.stderr[:300])
print("asr_top1_rerank_score occurrences:", s.count("asr_top1_rerank_score"),
      "| oof_score:", s.count("oof_score"))
