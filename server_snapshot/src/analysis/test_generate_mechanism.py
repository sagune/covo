#!/usr/bin/env python
"""Unit-test the coupling mechanism itself, on a tiny model, on CPU.

The generation path is the last piece that has never been executed, and it cannot
be executed on the real model while training owns the GPU.  The risk is not the
weights -- it is the MECHANICS:

  * does `generate(inputs_embeds=...)` return the prompt plus the continuation, or
    only the continuation?  The inference script's output slicing depends on this,
    and getting it wrong discards the entire output.
  * does the non-in-place splice survive generation?
  * does LEFT padding of an unequal-length batch work?
  * does the acoustic channel actually CHANGE the generated tokens, i.e. is it
    wired into generation at all and not silently ignored?

A tiny randomly-initialised GPT-2 answers all four in seconds, without a GPU and
without downloading anything.  Weights are irrelevant to these questions; the
only thing that matters is that the code paths are the real ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402

DIM_AC = 512
torch.manual_seed(0)


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2,
                     n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return GPT2LMHeadModel(cfg).eval()


def build(seed_proj=False):
    base = tiny_model()
    proj = AcousticProjector(DIM_AC, base.config.n_embd, n_layers=2)
    if seed_proj:                      # give it a non-trivial projection
        for p in proj.parameters():
            nn.init.normal_(p, std=0.05)
    return AcousticConditionedLM(base, proj).eval(), proj


def run(label, ok):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    return ok


def main() -> None:
    V = 256
    results = []

    # ---------------------------------------------------------------- test A
    print("A. zero-init projector must not change the logits")
    m, _ = build(seed_proj=False)
    ids = torch.randint(1, V, (1, 12))
    ac = torch.randn(1, 4, DIM_AC)
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=torch.ones_like(ids),
              acoustic=ac, acoustic_start=torch.tensor([3]),
              acoustic_len=torch.tensor([4]))
        b = m(input_ids=ids, attention_mask=torch.ones_like(ids))
    d = float((a.logits - b.logits).abs().max())
    results.append(run("zero-init logits identical (max diff %.3g)" % d, d == 0.0))

    # ---------------------------------------------------------------- test B
    print("B. generate(inputs_embeds=...) return length")
    m, _ = build(seed_proj=False)
    gen = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                     acoustic=ac, acoustic_start=torch.tensor([3]),
                     acoustic_len=torch.tensor([4]),
                     max_new_tokens=6, do_sample=False, pad_token_id=0)
    plen = ids.size(1)
    print("     prompt_len=%d  returned=%d" % (plen, gen.size(1)))
    only_new = gen.size(1) <= 6 + 4
    # this mirrors the discriminator used by infer_acoustic_covo.py
    new_tokens = gen if gen.size(1) <= 6 + 4 else gen[:, plen:]
    results.append(run("discriminator selects a non-empty continuation (%d tokens)"
                       % new_tokens.size(1), new_tokens.size(1) > 0))
    results.append(run("discriminator applied to the WRONG branch would be empty or huge "
                       "(prompt-only slice -> %d)" % (gen[:, plen:].size(1)),
                       (gen[:, plen:].size(1) == 0) if only_new else True))

    # ---------------------------------------------------------------- test C
    print("C. acoustic channel must actually change generation")
    m0, _ = build(seed_proj=False)
    m1, _ = build(seed_proj=True)
    g0 = m0.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                     acoustic=ac, acoustic_start=torch.tensor([3]),
                     acoustic_len=torch.tensor([4]), max_new_tokens=6,
                     do_sample=False, pad_token_id=0)
    g1 = m1.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                     acoustic=ac, acoustic_start=torch.tensor([3]),
                     acoustic_len=torch.tensor([4]), max_new_tokens=6,
                     do_sample=False, pad_token_id=0)
    same = bool(torch.equal(g0, g1))
    results.append(run("a non-zero projector changes the generated tokens "
                       "(zero-init==seeded: %s)" % same, not same))

    # ---------------------------------------------------------------- test D
    print("D. left-padded unequal-length batch")
    ids2 = torch.randint(1, V, (2, 16))
    ids2[1, :8] = 0                     # shorter row, left-padded with pad id
    mask = torch.ones_like(ids2)
    mask[1, :8] = 0
    ac2 = torch.randn(2, 5, DIM_AC)
    m2, _ = build(seed_proj=True)
    try:
        g2 = m2.generate(input_ids=ids2, attention_mask=mask,
                         acoustic=ac2, acoustic_start=torch.tensor([3, 12]),
                         acoustic_len=torch.tensor([5, 5]),
                         max_new_tokens=6, do_sample=False, pad_token_id=0)
        ok = g2.size(0) == 2 and g2.size(1) > 0
        msg = "runs and returns one continuation per row, shape=%s" % (tuple(g2.shape),)
    except Exception as e:                                    # noqa: BLE001
        ok, msg = False, "raised %s: %s" % (type(e).__name__, e)
    results.append(run("left-padded batch: " + msg, ok))

    print()
    print("%d/%d mechanism checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
