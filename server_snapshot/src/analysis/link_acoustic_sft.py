#!/usr/bin/env python
"""Link the SFT training rows to their acoustic vectors, and PROVE the pairing.

The acoustic index and the SFT file were produced from the same 17,301-row
evidence file, so they should correspond positionally -- but the SFT file keeps
only the SHARD-LOCAL `id` (0..999, repeated 18 times) and does not carry the
uttid, so the pairing cannot be checked with a key.  A silent off-by-one here
would attach the acoustics of one utterance to the text of another and would
look like a modelling failure rather than a data bug, so this script verifies
positionally by locating each acoustic row's `top1` string inside the SFT row's
user message, and refuses to write output if the match rate is not essentially
perfect.

Output adds two fields per row: `uttid` (from the acoustic index) and
`ac_row` (its row number in the acoustic index), so downstream code never has to
rely on ordering again.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The prompt renders the hypothesis without Chinese punctuation, while the raw
# top-1 we aligned acoustics against still carries it (22.9% of rows differ from
# their normalised form).  Comparing on punctuation-stripped text makes a genuine
# off-by-one visible instead of drowning it in ~1.3% of pure-rendering diffs.
_STRIP = re.compile(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9]+")


def strip(s: str) -> str:
    return _STRIP.sub("", s or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--ac-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-match", type=float, default=0.999)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    sft = [json.loads(l) for l in Path(a.sft).read_text(encoding="utf-8").splitlines() if l.strip()]
    ac = [json.loads(l) for l in Path(a.ac_index).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        sft, ac = sft[: a.limit], ac[: a.limit]
    print("sft rows: %d   acoustic rows: %d" % (len(sft), len(ac)))

    n = min(len(sft), len(ac))
    matched = 0
    matched_exact = 0
    first_bad = None
    bad_examples = []
    for i in range(n):
        top1 = ac[i].get("top1") or ""
        msgs = sft[i].get("messages") or []
        user = ""
        for m in msgs:
            if m.get("role") == "user":
                user = m.get("content") or ""
                break
        if top1 and top1 in user:
            matched_exact += 1
        if strip(top1) and strip(top1) in strip(user):
            matched += 1
        else:
            if first_bad is None:
                first_bad = i
            if len(bad_examples) < 5:
                bad_examples.append({
                    "row": i, "uttid": ac[i].get("id"),
                    "top1": top1[:60],
                    "top1_in_user": top1 in user,
                    "user_head": user[:200],
                })

    rate = matched / max(n, 1)
    print("positional match, punctuation-stripped: %d / %d = %.4f" % (matched, n, rate))
    print("positional match, byte-exact            : %d / %d = %.4f"
          % (matched_exact, n, matched_exact / max(n, 1)))
    if first_bad is not None:
        print("first unmatched row: %d" % first_bad)
        for b in bad_examples:
            print("  row %d uttid=%s top1_in_user=%s\n     top1=%s\n     user=%s"
                  % (b["row"], b["uttid"], b["top1_in_user"], b["top1"], b["user_head"]))

    if a.report_only:
        return
    if not a.limit and len(ac) != len(sft):
        raise SystemExit(
            "REFUSING to write: the acoustic index has %d rows but the row file has %d. "
            "A SHORT index means the export was still being written when this ran -- that "
            "silently truncates the linked file while still reporting a 1.0 match rate on "
            "the overlapping prefix (it did exactly that to the THCHS-30 eval set on "
            "2026-09-15, leaving 216 of 2495 rows). Wait for the export to finish."
            % (len(ac), len(sft)))
    if rate < a.min_match:
        raise SystemExit(
            "REFUSING to write: positional match rate %.4f < %.4f -- the acoustic "
            "rows do not line up with the SFT rows, so pairing by order is wrong."
            % (rate, a.min_match))

    out = []
    for i in range(n):
        row = dict(sft[i])
        row["uttid"] = ac[i].get("id")
        row["ac_row"] = i
        out.append(row)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out),
                           encoding="utf-8")
    print("wrote %d rows to %s (verified pairing)" % (len(out), a.out))


if __name__ == "__main__":
    main()
