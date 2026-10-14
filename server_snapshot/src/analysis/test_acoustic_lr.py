#!/usr/bin/env python
"""Verify the projector's separate learning rate actually applies.

The projector is 19M parameters trained from scratch while the LoRA is a fine-tune
of pretrained weights, so they need different LRs (LLaVA trains the projector orders
of magnitude above the LLM). Getting this wrong is subtle: training runs happily and
merely produces a weak channel, which reads as "acoustics do not help" rather than
as a bug.

Checks on a tiny GPT-2, CPU only:
  1. Trainer accepts an explicit optimiser (optimizers=(opt, None));
  2. the two parameter groups carry the two LRs;
  3. after a few steps the projector has moved noticeably MORE than it would have at
     the LLM's fine-tuning rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))
sys.path.insert(0, str(B / "scripts"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402

VOCAB, DIM_AC = 256, 512
results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def build():
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=64, n_layer=2,
                     n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    model = get_peft_model(GPT2LMHeadModel(cfg),
                           LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                      task_type="CAUSAL_LM", target_modules=["c_attn"]))
    proj = AcousticProjector(DIM_AC, 64, n_layers=2)
    return AcousticConditionedLM(model, proj), model, proj


def rows(n=4):
    out = []
    for _ in range(n):
        out.append({"input_ids": torch.randint(1, VOCAB, (12,)),
                    "attention_mask": torch.ones(12, dtype=torch.long),
                    "labels": torch.randint(1, VOCAB, (12,)),
                    "acoustic": torch.randn(3, DIM_AC),
                    "acoustic_start": torch.tensor(2),
                    "acoustic_len": torch.tensor(3)})
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, r):
        self.r = r

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        return self.r[i]


def main() -> None:
    from train_acoustic_covo import AcousticTrainer
    from transformers import TrainingArguments

    wrapper, model, proj = build()
    llm_lr, proj_lr = 2e-5, 3e-4
    proj_ids = {id(p) for p in proj.parameters()}
    llm_params = [p for p in wrapper.parameters() if p.requires_grad and id(p) not in proj_ids]
    proj_params = [p for p in proj.parameters() if p.requires_grad]
    optim = torch.optim.AdamW([{"params": llm_params, "lr": llm_lr},
                               {"params": proj_params, "lr": proj_lr}],
                              betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    print("1. the two parameter groups carry the two learning rates")
    lrs = [g["lr"] for g in optim.param_groups]
    check("LLM group lr == %.0e" % llm_lr, lrs[0] == llm_lr, str(lrs))
    check("projector group lr == %.0e" % proj_lr, lrs[1] == proj_lr, str(lrs))
    check("groups are disjoint", not ({id(p) for p in llm_params} & {id(p) for p in proj_params}))
    check("projector group is non-empty", len(proj_params) > 0, "%d tensors" % len(proj_params))

    before = {k: v.detach().clone() for k, v in proj.state_dict().items()
              if k.endswith("weight")}
    import tempfile
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), max_steps=6,
                             per_device_train_batch_size=2, save_strategy="no",
                             logging_steps=100, report_to="none",
                             remove_unused_columns=False)
    tr = AcousticTrainer(model=wrapper, args=args, train_dataset=DS(rows()),
                         data_collator=lambda b: {k: torch.stack([x[k] for x in b])
                                                  for k in b[0]},
                         optimizers=(optim, None))
    print("2. Trainer accepts the explicit optimiser")
    try:
        tr.train()
        check("Trainer.train() ran with optimizers=(opt, None)", True)
    except Exception as e:                                    # noqa: BLE001
        check("Trainer.train() ran with optimizers=(opt, None)", False,
              "%s: %s" % (type(e).__name__, e))
        print()
        print("%d/%d checks pass" % (sum(results), len(results)))
        raise SystemExit(1)

    print("3. the projector moved at the higher rate")
    after = proj.state_dict()
    moved = max(float((after[k].float() - before[k].float()).abs().max()) for k in before)
    # 6 steps at 3e-4 can move a weight by ~1.8e-3; at 2e-5 only ~1.2e-4
    check("projector moved by %.2e after 6 steps" % moved, moved > 5e-4,
          "at the fine-tuning LR 6 steps would give ~1.2e-4")

    print()
    print("%d/%d checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
