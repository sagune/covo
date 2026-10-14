#!/usr/bin/env python
"""Prove the acoustic trainer's TEXT interface matches the baseline trainer.

The whole comparison rests on the acoustic run differing from the text-only run in
exactly ONE respect: the acoustic channel.  If the two trainers render or tokenize
the prompt differently, the experiment has two variables and no result is
attributable to the acoustics.

The baseline (covo/scripts/train_lora_sft.py `_load_texts` + `tokenize`) does:

    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False,
                                         enable_thinking=False)
    ids  = tokenizer(text, max_length=M, truncation=True, padding=False)["input_ids"]

with the tokenizer's DEFAULT add_special_tokens=True, whereas
acoustic_conditioning.build_acoustic_inputs passes add_special_tokens=False.  This
script compares the two tokenisations row by row, so a difference is caught now
rather than showing up as an unexplained result.

CPU-only: needs the tokenizer, not the model, so it runs while training owns the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticStore, build_acoustic_inputs  # noqa: E402


def render_baseline(tokenizer, messages, max_length):
    kw = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
    try:
        text = tokenizer.apply_chat_template(messages, **kw)
    except TypeError:
        kw.pop("enable_thinking")
        text = tokenizer.apply_chat_template(messages, **kw)
    text = text[: max(1, int(max_length) * 8)]
    return tokenizer(text, max_length=int(max_length), truncation=True, padding=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    rows = [json.loads(l) for l in Path(a.rows).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: a.limit]
    store = AcousticStore(a.bin, a.index)

    same = 0
    diff = []
    for r in rows:
        msgs = r.get("messages") or []
        base = render_baseline(tok, msgs, a.max_length)
        mine = build_acoustic_inputs(tok, msgs, int(r["ac_row"]), store,
                                     max_length=a.max_length, disable_thinking=True,
                                     train=True)["input_ids"]
        if base == mine:
            same += 1
        elif len(diff) < 5:
            # locate the first divergence for a readable report
            k = next((i for i in range(min(len(base), len(mine))) if base[i] != mine[i]),
                     min(len(base), len(mine)))
            diff.append((r.get("id"), len(base), len(mine), k,
                         tok.decode(base[max(0, k - 4):k + 4]),
                         tok.decode(mine[max(0, k - 4):k + 4])))

    print("rows compared            : %d" % len(rows))
    print("identical tokenisations  : %d" % same)
    print("DIFFERENT                : %d" % (len(rows) - same))
    for d in diff:
        print("   id=%s len base=%d mine=%d first diff @%d\n      base=%r\n      mine=%r" % d)
    print()
    if same == len(rows) and rows:
        print("VERDICT: PASS -- the text interface is identical; the only difference "
              "between the acoustic and text-only runs is the acoustic channel.")
        raise SystemExit(0)
    print("VERDICT: FAIL -- the prompt/tokenisation differs from the baseline, so the "
          "comparison would be confounded by two variables.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
