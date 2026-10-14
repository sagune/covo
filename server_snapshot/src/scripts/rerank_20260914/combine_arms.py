#!/usr/bin/env python3
"""Combine a GENERATOR arm with a SELECTOR arm, filling in only where the generator abstained.

Why this exists.  RESULTS 10.12 (deployable policy) shows the shipped corrector on ST-CMDS
is precise but timid: edit precision 78.5%, edit rate only 6.7%, coverage 13.4% - it
realises 9.5% of the 1655-character editable headroom.  Meanwhile AISHELL dev proves the
same model family CAN edit 28.0% of rows at 75.2% precision.  So the failure is
ABSTENTION, not inaccuracy.

A selector arm (the likelihood scorer) never abstains: it always returns a candidate, so
its coverage is high by construction.  This script therefore keeps the generator's output
wherever the generator actually changed the text - preserving its editing, which is 57% of
its value - and substitutes the selector's choice only on rows the generator left equal to
the front-end top-1.

    combine_arms.py --generator <preds> --selector <preds> --out <preds>

This is a combination ANALYSIS, not a new trained method: it is reported alongside both
input arms and its editing side is inherited wholesale from the generator.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--selector", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    G, S = load(a.generator), load(a.selector)
    if len(G) != len(S):
        print("row count differs: %d vs %d" % (len(G), len(S)))
    n = min(len(G), len(S))
    rows, stats = [], {"abstained_used_selector": 0, "abstained_selector_same": 0,
                       "kept_generator_edit": 0, "selector_missing": 0}
    for i in range(n):
        g, s = G[i], S[i]
        top = norm((g.get("input") or {}).get("asr_top1") or "")
        gout = norm(g.get("prediction") or "")
        sout = norm(s.get("prediction") or "")
        r = dict(g)
        if gout and gout != top:
            stats["kept_generator_edit"] += 1
        elif sout:
            if sout != gout:
                stats["abstained_used_selector"] += 1
            else:
                stats["abstained_selector_same"] += 1
            r["prediction"] = s.get("prediction")
            r["combined_from"] = "selector"
        else:
            stats["selector_missing"] += 1
        rows.append(r)

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    print("rows %d -> %s" % (len(rows), a.out))
    for k, v in stats.items():
        print("  %-26s %d (%.1f%%)" % (k, v, 100.0 * v / max(1, len(rows))))


if __name__ == "__main__":
    main()
