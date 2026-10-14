#!/usr/bin/env python3
"""Map a selector model's `{"choice": k}` output back to candidate text.

Completes the candidate-selector ablation: the model is asked to emit the index of
the candidate it prefers, this resolves that index against the numbered list that
was actually shown, and writes an ordinary predictions file so restore_eval.py and
every downstream metric work unchanged.

The list is parsed out of the selector prompt itself, so the mapping cannot drift
from what the model saw.  Unparseable outputs fall back to the front-end top-1 and
are counted, because a selector that breaks the schema is a failure mode the
generator does not have and the ablation must show it.
"""
import argparse
import json
import re
from pathlib import Path

sys_path = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
import sys
sys.path.insert(0, sys_path)
from covo.text import normalize_chinese_text as norm  # noqa: E402

LINE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+\|\s")
CHOICE = re.compile(r'"choice"\s*:\s*(\d+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="the selector prompt file (numbered list)")
    ap.add_argument("--predictions", required=True, help="raw selector output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lists = {}
    for l in Path(a.messages).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        msgs = r.get("messages") or []
        user = next((m.get("content") for m in msgs if m.get("role") == "user"), "") or ""
        cands = {}
        for line in user.splitlines():
            m = LINE.match(line)
            if m:
                cands[int(m.group(1))] = m.group(2).strip()
        lists[str(r.get("id"))] = (cands, norm(((r.get("input") or {}).get("asr_top1")) or ""))

    total = ok = bad = miss = 0
    rows = []
    for l in Path(a.predictions).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        total += 1
        cands, top1 = lists.get(str(r.get("id")), ({}, ""))
        raw = r.get("prediction")
        if isinstance(raw, dict):
            txt = json.dumps(raw, ensure_ascii=False)
        else:
            txt = str(raw or "")
        m = CHOICE.search(txt)
        if m and int(m.group(1)) in cands:
            chosen = cands[int(m.group(1))]
            ok += 1
        else:
            chosen = top1
            if m:
                miss += 1
            else:
                bad += 1
        nr = json.loads(json.dumps(r, ensure_ascii=False))
        nr["prediction"] = chosen
        nr["selector_raw"] = txt[:120]
        rows.append(nr)

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    print("rows %d | parsed and in range %d (%.1f%%) | index out of range %d | schema failure %d"
          % (total, ok, 100.0 * ok / max(1, total), miss, bad))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
