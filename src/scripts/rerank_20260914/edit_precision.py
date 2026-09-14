#!/usr/bin/env python3
"""Edit precision vs coverage: WHEN the backend edits, is it right, and how much of the
editable space does it reach?

The anatomy (RESULTS 10.1) shows ST-CMDS is 93.3% no_change.  That alone cannot say
whether the model is *conservative but accurate* (raise its willingness to edit and it
wins) or *reckless* (it should be told to stop).  Those need opposite remedies, so this
separates them:

  precision = of the edits that changed the error count, how many reduced it
  coverage  = of the rows where the prompt showed a better candidate, how many were fixed

Conservative-but-accurate shows high precision with low coverage; reckless shows low
precision.  The two are reported with the raw headroom in characters so the size of the
prize is visible next to the hit rate.

    edit_precision.py --records <predictions.jsonl> --label "..."
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402
from covo.metrics import edit_distance                  # noqa: E402


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
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    n = edited = helped = hurt = tie = 0
    worthy = worthy_fixed = 0
    worthy_chars = fixed_chars = 0
    hurt_chars = 0
    for line in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        out = norm(r.get("prediction") or "")
        if not ref or not top:
            continue
        n += 1
        e0 = edit_distance(list(ref), list(top))
        e1 = edit_distance(list(ref), list(out)) if out else e0
        if out != top:
            edited += 1
            if e1 < e0:
                helped += 1
            elif e1 > e0:
                hurt += 1
                hurt_chars += e1 - e0
            else:
                tie += 1
        vis = set(texts_of(inp.get("nbest")))
        if vis:
            best = min(edit_distance(list(ref), list(t)) for t in vis)
            if best < e0:
                worthy += 1
                worthy_chars += e0 - best
                if e1 < e0:
                    worthy_fixed += 1
                    fixed_chars += e0 - e1

    decisive = helped + hurt
    print("# %s   rows=%d" % (a.label or a.records, n))
    print("  edited rows                : %d (%.1f%% of all rows)" % (edited, 100.0 * edited / max(1, n)))
    print("  of those  helped/hurt/tie  : %d / %d / %d" % (helped, hurt, tie))
    print("  EDIT PRECISION             : %.1f%%  (helped / (helped+hurt))"
          % (100.0 * helped / max(1, decisive)))
    print("  characters gained by edits : +%d" % (fixed_chars - hurt_chars))
    print("     from helped edits       : +%d" % fixed_chars)
    print("     lost to hurt edits      : -%d" % hurt_chars)
    print("  EDITABLE ROWS (the prompt showed a strictly better candidate): %d (%.1f%%)"
          % (worthy, 100.0 * worthy / max(1, n)))
    print("  COVERAGE                   : %.1f%%  (%d of %d editable rows improved)"
          % (100.0 * worthy_fixed / max(1, worthy), worthy_fixed, worthy))
    print("  headroom on editable rows  : %d characters" % worthy_chars)
    print("  net vs the front end       : %+.1f%% of the editable headroom realised"
          % (100.0 * (fixed_chars - hurt_chars) / max(1, worthy_chars)))


if __name__ == "__main__":
    main()
