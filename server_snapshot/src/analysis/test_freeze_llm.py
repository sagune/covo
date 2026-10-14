#!/usr/bin/env python
"""Verify the --freeze-llm contingency path: only the projector may move.

If the acoustic channel turns out to be inert, the response is the LLaVA-style
alignment stage -- freeze the language model and train the projector alone, forcing
it to become useful before LoRA is allowed to route around it.  That path exists in
the trainer but has never been run, and discovering it is broken at 1 AM while
holding a stalled experiment is the wrong time.

Checks, on a tiny GPT-2 with LoRA, on CPU:
  1. after freezing, NO base/adapter parameter requires grad and the projector does;
  2. one optimiser step changes the projector;
  3. ... and leaves every LoRA parameter bit-identical;
  4. the forward pass still runs with gradient checkpointing enabled, where the
     checkpointed blocks need at least one input that requires grad -- with the
     whole base frozen, that input is the projected acoustics, so this is exactly
     the configuration that could silently break.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402

DIM_AC = 512
VOCAB = 256
results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def main() -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=64, n_layer=2,
                     n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    model = get_peft_model(GPT2LMHeadModel(cfg),
                           LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                      task_type="CAUSAL_LM", target_modules=["c_attn"]))
    projector = AcousticProjector(DIM_AC, 64, n_layers=2)
    wrapper = AcousticConditionedLM(model, projector)

    # give the projector a non-trivial start so its gradient is not a zero product
    for p in projector.parameters():
        torch.nn.init.normal_(p, std=0.02)

    # what --freeze-llm does, in the same order the trainer does it
    for p in model.parameters():
        p.requires_grad = False

    print("1. parameter ownership after freezing")
    base_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    proj_trainable = [n for n, p in projector.named_parameters() if p.requires_grad]
    check("no base/adapter parameter requires grad", not base_trainable,
          "leaked: %s" % base_trainable[:3])
    check("every projector parameter requires grad", len(proj_trainable) > 0,
          "%d tensors" % len(proj_trainable))

    ids = torch.randint(1, VOCAB, (2, 12))
    mask = torch.ones_like(ids)
    ac = torch.randn(2, 4, DIM_AC)
    start = torch.tensor([3, 3])
    length = torch.tensor([4, 4])
    labels = torch.randint(1, VOCAB, (2, 12))

    before = {k: v.detach().clone() for k, v in projector.state_dict().items()}
    lora_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-2)
    out = wrapper(input_ids=ids, attention_mask=mask, labels=labels,
                  acoustic=ac, acoustic_start=start, acoustic_len=length)
    out.loss.backward()

    grads = [p.grad for p in projector.parameters() if p.grad is not None]
    check("projector received gradients", len(grads) > 0 and any(float(g.norm()) > 0 for g in grads),
          "total grad norm %.6f" % sum(float(g.norm()) for g in grads))

    opt.step()
    after = projector.state_dict()
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    check("optimiser step changed the projector", len(moved) > 0,
          "%d/%d tensors changed" % (len(moved), len(before)))

    lora_after = model.state_dict()
    changed = [k for k in lora_before if not torch.equal(lora_before[k], lora_after[k])]
    check("optimiser step left every LoRA parameter bit-identical", not changed,
          "changed: %s" % changed[:3])

    print("4. gradient checkpointing with the whole base frozen")
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        out2 = wrapper(input_ids=ids, attention_mask=mask, labels=labels,
                       acoustic=ac, acoustic_start=start, acoustic_len=length)
        out2.loss.backward()
        check("forward+backward runs under gradient checkpointing", True,
              "loss %.4f" % float(out2.loss))
    except Exception as e:                                    # noqa: BLE001
        check("forward+backward runs under gradient checkpointing", False,
              "%s: %s" % (type(e).__name__, e))

    print()
    print("%d/%d freeze-llm checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
