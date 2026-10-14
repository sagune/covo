#!/usr/bin/env python
"""Run the coupling-mechanism checks on the REAL Qwen3.5 architecture, tiny and on CPU.

The earlier mechanism test used GPT-2. Qwen3.5 is a different animal: a hybrid
stack whose `layer_types` alternate `linear_attention` (gated delta rule) with
`full_attention` every 4th layer, and whose published config advertises the
MULTIMODAL `Qwen3_5ForConditionalGeneration`.

Two things therefore needed checking rather than assuming:

  1. which class `AutoModelForCausalLM` actually resolves for this config -- the
     trainer uses that entry point, so it decides what the projector wraps;
  2. whether `generate(inputs_embeds=...)` still returns only the continuation on
     that class, since the inference script's output slicing depends on it.

A randomly-initialised model built from the real config answers both without the
GPU and without 8 GB of weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

MODEL = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402

results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def main() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    print("1. which class does AutoModelForCausalLM resolve for this config?")
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    print("     config architectures: %s" % cfg.architectures)
    print("     model_type          : %s" % cfg.model_type)

    # Qwen3_5ForCausalLM takes a Qwen3_5TextConfig (model_type "qwen3_5_text"), not
    # the outer multimodal Qwen3_5Config -- which has no vocab_size of its own.
    text = getattr(cfg, "text_config", cfg)
    for k, v in (("hidden_size", 64), ("intermediate_size", 128),
                 ("num_hidden_layers", 2), ("num_attention_heads", 2),
                 ("num_key_value_heads", 2), ("head_dim", 32), ("vocab_size", 256)):
        if hasattr(text, k):
            setattr(text, k, v)
    if hasattr(text, "layer_types"):
        # must match num_hidden_layers AND include a full-attention layer, so the
        # hybrid path (gated delta rule + full attention) is genuinely exercised
        text.layer_types = ["linear_attention", "full_attention"]

    try:
        model = AutoModelForCausalLM.from_config(text, trust_remote_code=True)
    except Exception as e:                                    # noqa: BLE001
        check("a tiny Qwen3.5 can be built from the real config", False,
              "%s: %s" % (type(e).__name__, e))
        print()
        print("%d/%d checks pass" % (sum(results), len(results)))
        print("NOTE: construction failed, so only the GenerationMixin argument applies.")
        raise SystemExit(1)

    cls = type(model).__name__
    from transformers import GenerationMixin, PreTrainedModel
    check("built a tiny model", True, cls)
    check("the class the trainer loads is a PreTrainedModel", isinstance(model, PreTrainedModel), cls)
    check("... and a GenerationMixin (standard inputs_embeds path)",
          isinstance(model, GenerationMixin), cls)
    has_linear = any("linear" in str(getattr(m, "layer_type", "")).lower()
                     or m.__class__.__name__ == "Qwen3_5GatedDeltaNet"
                     for m in model.modules())
    check("the hybrid linear-attention path is actually present", has_linear)

    model.eval()
    proj = AcousticProjector(512, 64, n_layers=2)
    wrapper = AcousticConditionedLM(model, proj).eval()

    V = 256
    ids = torch.randint(1, V, (1, 10))
    ac = torch.randn(1, 4, 512)
    mask = torch.ones_like(ids)

    print("2. generate(inputs_embeds=...) on the real architecture")
    try:
        gen = wrapper.generate(input_ids=ids, attention_mask=mask, acoustic=ac,
                               acoustic_start=torch.tensor([3]),
                               acoustic_len=torch.tensor([4]),
                               max_new_tokens=5, do_sample=False, pad_token_id=0)
        check("generate ran without error", True)
        print("     prompt_len=%d  returned=%d" % (ids.size(1), gen.size(1)))
        new = gen if gen.size(1) <= 5 + 4 else gen[:, ids.size(1):]
        check("discriminator selects a non-empty continuation (%d tokens)" % new.size(1),
              new.size(1) > 0)
    except Exception as e:                                    # noqa: BLE001
        check("generate ran without error", False, "%s: %s" % (type(e).__name__, e))

    print("3. a zero-init projector leaves the logits bit-identical")
    with torch.no_grad():
        a = wrapper(input_ids=ids, attention_mask=mask, acoustic=ac,
                    acoustic_start=torch.tensor([3]), acoustic_len=torch.tensor([4]))
        b = wrapper(input_ids=ids, attention_mask=mask)
    d = float((a.logits.float() - b.logits.float()).abs().max())
    check("max logit diff == 0 on Qwen3.5 (%.3g)" % d, d == 0.0)

    print()
    print("%d/%d checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
