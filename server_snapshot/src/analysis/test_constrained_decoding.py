#!/usr/bin/env python
"""Unit-test the SEAL-style constrained decoder (CPU only, no model).

A character-level fake tokenizer makes every masking decision directly checkable:
token id == ord(char), so an allowed set of {"A","B"} must leave exactly those ids
(run: "...") finite and every other id at -inf.

The cases that matter are not the happy path but the RELEASE conditions -- the
constraint must never be able to trap the model in an unfinishable sequence, so it
has to switch itself off outside the text value, after the value closes, and once
the sentence has run past the hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from constrained_decoding import (  # noqa: E402
    AcousticConstrainedLogitsProcessor, build_char_to_ids, load_allowed_sets,
)

VOCAB = 256
NEG = float("-inf")


class FakeTok:
    """Character-level tokenizer: id == ord(char)."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if 0 <= int(i) < VOCAB)


def ids_of(s):
    return [ord(c) for c in s]


def finite(scores):
    return {int(i) for i in torch.nonzero(torch.isfinite(scores)).flatten().tolist()}


results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def make(allowed_sets, first_extra=None):
    chars = [c for row in allowed_sets for st in row for c in st]
    tok = FakeTok()
    return AcousticConstrainedLogitsProcessor(
        allowed_sets, build_char_to_ids(tok, chars), tok,
        eos_token_id=0, extra_allowed_ids=[ord('"'), ord('}')],
        first_step_extra_ids=first_extra)


def main() -> None:
    print("1. outside the text value -> no masking at all")
    p = make([[["A", "B"]]])
    x = torch.randn(1, VOCAB)
    out = p(torch.tensor([ids_of('{"te')]), x.clone())
    check("scores untouched before the marker", torch.allclose(out, x))

    print("2. inside the value at position 0 -> only the allowed set survives")
    p = make([[["A", "B"]]])
    x = torch.randn(1, VOCAB)
    out = p(torch.tensor([ids_of('{"text":"')]), x.clone())
    got = finite(out[0])
    want = {ord("A"), ord("B"), ord('"'), ord("}"), 0}
    check("mask leaves exactly {A,B,quote,brace,eos}", got == want,
          "got=%s" % sorted(got))

    print("3. position advances with each character emitted")
    p = make([[["A", "B"], ["C", "D"]]])
    out = p(torch.tensor([ids_of('{"text":"A')]), torch.randn(1, VOCAB))
    got = finite(out[0])
    check("second position uses set 1, not set 0",
          got == {ord("C"), ord("D"), ord('"'), ord("}"), 0}, "got=%s" % sorted(got))

    print("4. value closed -> constraint released")
    p = make([[["A", "B"]]])
    x = torch.randn(1, VOCAB)
    out = p(torch.tensor([ids_of('{"text":"AB"}')]), x.clone())
    check("scores untouched after the closing quote", torch.allclose(out, x))

    print("5. sentence ran past the hypothesis -> constraint released")
    p = make([[["A"]]])
    x = torch.randn(1, VOCAB)
    out = p(torch.tensor([ids_of('{"text":"AB')]), x.clone())
    check("scores untouched past the end of the hypothesis", torch.allclose(out, x))

    print("6. a batch is masked row by row, independently")
    p = make([[["A"]], [["Z"]]])
    out = p(torch.tensor([ids_of('{"text":"'), ids_of('{"text":"')]), torch.randn(2, VOCAB))
    g0, g1 = finite(out[0]), finite(out[1])
    check("row0 allows A only", ord("A") in g0 and ord("Z") not in g0, "row0=%s" % sorted(g0))
    check("row1 allows Z only", ord("Z") in g1 and ord("A") not in g1, "row1=%s" % sorted(g1))

    print("7. last marker wins (the prompt may itself contain an example)")
    p = make([[["A", "B"], ["C", "D"]]])
    txt = '示例 {"text":"X"} 现在:{"text":"A'
    out = p(torch.tensor([ids_of(txt)]), torch.randn(1, VOCAB))
    got = finite(out[0])
    check("uses the LAST marker, so position 1 is active",
          got == {ord("C"), ord("D"), ord('"'), ord("}"), 0}, "got=%s" % sorted(got))

    print("8. load_allowed_sets builds hypothesis-first sets")
    import json
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "sets.jsonl"
    tmp.write_text(json.dumps({"id": "u1", "top1": "好英",
                               "topk": [["郝", "号"], ["音"]]}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    sets = load_allowed_sets(str(tmp), top_k=5)
    check("hypothesis char is first, alternatives follow",
          sets["u1"] == [["好", "郝", "号"], ["英", "音"]], str(sets["u1"]))
    check("missing alternatives still yield a length-1 set",
          all(len(s) >= 1 for s in sets["u1"]))

    print("9. a multi-token character must RELEASE, not trap, the model")
    # The mask admits only a character's FIRST token.  If an allowed character
    # needs more than one token, the emitted token decodes to an incomplete
    # character, the position never advances, and the model is masked into
    # re-emitting a token it cannot complete.  10.7% of the distinct characters in
    # the real ST-CMDS sets are multi-token, so this path matters.
    x = torch.randn(1, VOCAB)
    p = make([[["A", "B"]]])
    p.safe_chars = {"A"}                     # pretend "B" is multi-token
    out = p(torch.tensor([ids_of('{"text":"')]), x.clone())
    check("released when an allowed char is multi-token", torch.allclose(out, x))

    p2 = make([[["A", "B"]]])
    p2.safe_chars = {"A", "B"}
    out2 = p2(torch.tensor([ids_of('{"text":"')]), x.clone())
    check("still masked when every allowed char is single-token",
          not torch.allclose(out2, x))

    print()
    print("%d/%d checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
