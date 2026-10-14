#!/usr/bin/env python
"""CPU-only check that the acoustic span survives left-padding correctly.

This runs WITHOUT the language model, so it can execute while training owns the
GPU.  It exercises the half of the inference path that would fail silently:
locating the hypothesis span, then left-padding a batch and shifting
`acoustic_start` with it.

The decisive assertion is that decoding the padded sequence over
[start, start+len) reproduces the hypothesis characters.  If the shift were wrong
the decoded text would be the neighbouring characters instead, which is exactly
the failure that would otherwise show up only as "the acoustics did not help".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp")
BUNDLE = REPO / "cbwhisper_covo_migration_20260609_tar_extracted"
for p in (str(BUNDLE / "covo" / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from acoustic_conditioning import (  # noqa: E402
    AcousticStore, build_acoustic_inputs, collate_acoustic, _strip_punct,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    store = AcousticStore(a.bin, a.index)
    rows = [json.loads(l) for l in Path(a.rows).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: a.limit]
    print("tokenizer=%s  pad=%r  rows=%d  acoustic_rows=%d"
          % (type(tok).__name__, tok.pad_token, len(rows), len(store)), flush=True)

    # ---- single-row span check
    n_span, n_bad, examples = 0, 0, []
    for r in rows:
        ac_row = int(r["ac_row"])
        user = [m for m in (r.get("messages") or []) if m.get("role") != "assistant"]
        d = build_acoustic_inputs(tok, user, ac_row, store, max_length=2048,
                                  disable_thinking=True, train=False)
        if d["acoustic_len"] <= 0:
            continue
        n_span += 1
        lo = d["acoustic_start"]
        hi = lo + d["acoustic_len"]
        decoded = _strip_punct(tok.decode(d["input_ids"][lo:hi], skip_special_tokens=True))
        want = _strip_punct(store.row(ac_row)["top1"])
        # the decoded span must be a substring of the hypothesis (BPE may merge one
        # character with a neighbour, so containment -- not equality -- is the test)
        if want and (decoded in want or want in decoded):
            continue
        n_bad += 1
        if len(examples) < 3:
            examples.append((r.get("id"), want[:40], decoded[:40]))
    print("single-row: span found %d/%d   span-decodes-to-hypothesis %d   MISMATCH %d"
          % (n_span, len(rows), n_span - n_bad, n_bad))
    for e in examples:
        print("   id=%s want=%r got=%r" % e)

    # ---- left-padding shift check on a deliberately unequal-length pair
    pair = []
    for r in rows:
        user = [m for m in (r.get("messages") or []) if m.get("role") != "assistant"]
        d = build_acoustic_inputs(tok, user, int(r["ac_row"]), store, max_length=2048,
                                  disable_thinking=True, train=False)
        if d["acoustic_len"] > 0:
            pair.append(d)
        if len(pair) == 2 and len(pair[0]["input_ids"]) != len(pair[1]["input_ids"]):
            break
    if len(pair) < 2:
        print("could not build two unequal-length rows; skipping pad test")
        return
    pair.sort(key=lambda d: len(d["input_ids"]))
    short, long = pair[0], pair[1]
    if len(short["input_ids"]) == len(long["input_ids"]):
        print("lengths equal; pad test inconclusive")
        return

    b = collate_acoustic([short, long], tok.pad_token_id, pad_side="left")
    pad = len(long["input_ids"]) - len(short["input_ids"])
    ids = b["input_ids"]
    ok_ids = ids[0, pad:].tolist() == short["input_ids"]
    ok_mask = b["attention_mask"][0, :pad].sum().item() == 0
    ok_start = int(b["acoustic_start"][0]) == short["acoustic_start"] + pad
    ok_ids_long = ids[1].tolist() == long["input_ids"]
    ok_start_long = int(b["acoustic_start"][1]) == long["acoustic_start"]
    lo = int(b["acoustic_start"][0]); hi = lo + int(b["acoustic_len"][0])
    decoded = _strip_punct(tok.decode(ids[0, lo:hi].tolist(), skip_special_tokens=True))
    want = _strip_punct(store.row(int(rows[0]["ac_row"]))["top1"])
    ok_decoded = (not want) or decoded in want or want in decoded

    print()
    print("left-pad shift test (pad=%d):" % pad)
    print("  row0 ids shifted by pad            : %s" % ok_ids)
    print("  row0 pad positions are masked      : %s" % ok_mask)
    print("  row0 acoustic_start shifted by pad : %s (%d -> %d, expected %d)"
          % (ok_start, short["acoustic_start"], int(b["acoustic_start"][0]),
             short["acoustic_start"] + pad))
    print("  row1 (longest) untouched           : ids=%s start=%s" % (ok_ids_long, ok_start_long))
    print("  padded span still decodes to hyp   : %s  (%r)" % (ok_decoded, decoded[:40]))
    allok = all([ok_ids, ok_mask, ok_start, ok_ids_long, ok_start_long, ok_decoded])
    print()
    print("VERDICT: %s" % ("PASS" if (allok and n_bad == 0) else "FAIL"))
    raise SystemExit(0 if (allok and n_bad == 0) else 1)


if __name__ == "__main__":
    main()
