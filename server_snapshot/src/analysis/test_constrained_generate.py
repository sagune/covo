#!/usr/bin/env python
"""Integration test: the constrained decoder under the REAL generation input pattern.

The decisive fact this test encodes, discovered by instrumenting a live `generate`:
under `inputs_embeds` generation the logits processor receives ONLY the generated
tokens -- the sequence starts EMPTY (shape (1, 0)) and the prompt is never present.
So the value marker can only come from the model's own output.

Two consequences worth testing rather than assuming:

  * the constraint must engage at exactly the step the model finishes emitting
    `"text":"`, and must stay off before that;
  * the marker must be matched TOLERANTLY. If the model emits `"text": "` with a
    space and we required `"text":"` exactly, the constraint would silently never
    engage and the arm would be indistinguishable from the unconstrained one.

A tiny model with Qwen's vocabulary runs the REAL tokenizer and processor, so ids line
up. Random weights are irrelevant -- the question is whether the constraint engages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
MODEL = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402
from constrained_decoding import (  # noqa: E402
    AcousticConstrainedLogitsProcessor, build_char_to_ids,
)

results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def finite_count(out):
    return int(torch.isfinite(out).sum())


def main() -> None:
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    V = len(tok)
    print("tokenizer vocab: %d" % V)

    allowed = [[c for c in "好英"], [c for c in "雄武"]]
    chars = [c for st in allowed for c in st]
    c2i = build_char_to_ids(tok, chars)
    extras = []
    for ch in ('"', "}", "\n", " "):
        extras.extend(tok.encode(ch, add_special_tokens=False))

    def make():
        return AcousticConstrainedLogitsProcessor(
            [allowed], c2i, tok, eos_token_id=tok.eos_token_id,
            extra_allowed_ids=extras, safe_chars=chars)

    proc = make()
    scores = torch.randn(1, V)

    print("1. the constraint is OFF until the value opens, then ON")
    # exactly the shapes generate passes: (1,0), then growing generated tokens
    closed = tok('{"text":', add_special_tokens=False)["input_ids"]
    opened = tok('{"text":"', add_special_tokens=False)["input_ids"]
    seqs = [
        ([], "empty (first generation step)"),
        (closed, "closed quoted key, value not yet open"),
        (opened, "value just opened"),
    ]
    for ids, desc in seqs:
        out = proc(torch.tensor([ids]), scores.clone())
        n = finite_count(out)
        print("     %-40s finite=%d" % (desc, n))
        if desc.startswith("value just opened"):
            check("masked once the value opens", n < V, "%d of %d" % (n, V))
        else:
            check("unmasked while outside the value (%s)" % desc, n == V)

    print("2. position advances with each character emitted")
    for extra_chars, want_second in ((1, True),):
        ids = opened + tok.encode("好", add_special_tokens=False)
        out = proc(torch.tensor([ids]), scores.clone())
        kept = {int(i) for i in torch.nonzero(torch.isfinite(out[0])).flatten().tolist()}
        want = set(c2i["雄"]) | set(c2i["武"]) | set(extras) | {tok.eos_token_id}
        # position 1's set is 雄武; position 0's is 好英
        check("second position uses the SECOND allowed set",
              set(c2i["雄"]) <= kept and not (set(c2i["好"]) <= kept),
              "kept=%d" % len(kept))

    print("3. whitespace-tolerant marker (the silent-disable failure)")
    for variant, desc in (('{"text": "', 'space after the colon'),
                          ('{ "text":"', 'space after the brace'),
                          ('{"text":"', 'canonical')):
        p = make()
        ids = tok(variant, add_special_tokens=False)["input_ids"]
        out = p(torch.tensor([ids]), scores.clone())
        check("engages for %s (%r)" % (desc, variant), finite_count(out) < V)

    print("4. released after the value closes")
    ids = tok('{"text":"好英"', add_special_tokens=False)["input_ids"]
    out = make()(torch.tensor([ids]), scores.clone())
    check("unmasked after the closing quote", finite_count(out) == V)

    print("5. a real generate call runs with the processor attached")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=V, n_positions=256, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=0, eos_token_id=0, pad_token_id=0)
    model = GPT2LMHeadModel(cfg).eval()
    proj = AcousticProjector(512, 64, n_layers=2)
    for q in proj.parameters():
        torch.nn.init.normal_(q, std=0.02)
    wrapper = AcousticConditionedLM(model, proj).eval()
    ids = torch.tensor([opened])
    try:
        gen = wrapper.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               acoustic=torch.randn(1, 2, 512),
                               acoustic_start=torch.tensor([2]),
                               acoustic_len=torch.tensor([2]),
                               max_new_tokens=4, do_sample=False,
                               pad_token_id=tok.pad_token_id, logits_processor=[make()])
        check("generate returned", True, "shape=%s" % (tuple(gen.shape),))
        # the prompt is invisible to the processor, so the first generated tokens are
        # the value's start only if the model itself emits the marker; a random model
        # will not, which is why check 1 above is the real assertion.
    except Exception as e:                                    # noqa: BLE001
        check("generate returned", False, "%s: %s" % (type(e).__name__, e))

    print()
    print("%d/%d integration checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
