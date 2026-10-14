#!/usr/bin/env python
"""Verify the checkpoint round-trip P4/P5 depend on, using a tiny model on CPU.

P4 loads NINE Trainer-saved checkpoints with `PeftModel.from_pretrained(base, ckpt)`
and P5 loads the final one twice.  If Trainer's checkpoint writer and my
`AcousticConditionedLM` wrapper disagree about what to write -- the wrapper is a
plain nn.Module around a PeftModel, which Trainer has never seen -- then every one
of those loads fails and the whole night's evaluation produces nothing.

Training itself was already proven by the pilot; SAVING was not, because the pilot
ran with `--save-steps 100000`.  This reproduces the exact save/load contract on a
randomly-initialised GPT-2, which costs seconds and needs no GPU.

It checks, in order:
  1. Trainer runs with the wrapper and saves a checkpoint at all.
  2. The checkpoint holds a LoRA adapter (adapter_config.json + weights).
  3. The SaveProjector callback wrote acoustic_projector.pt next to it.
  4. `PeftModel.from_pretrained(base, ckpt)` loads it without error.
  5. The projector state_dict round-trips and matches what was saved.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))
sys.path.insert(0, str(B / "scripts"))

from acoustic_conditioning import AcousticConditionedLM, AcousticProjector  # noqa: E402

DIM_AC = 512
VOCAB = 256
NEG = float("-inf")
results = []


def check(label, cond, extra=""):
    results.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label, (" -- " + extra) if extra else ""))


def tiny():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=64, n_layer=2,
                     n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return GPT2LMHeadModel(cfg)


def main() -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TrainingArguments
    from train_acoustic_covo import AcousticTrainer

    tmp = Path(tempfile.mkdtemp(prefix="ckpt_rt_"))
    print("workdir: %s" % tmp)

    base = tiny()
    peft_cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM", target_modules=["c_attn"])
    model = get_peft_model(base, peft_cfg)
    projector = AcousticProjector(DIM_AC, int(model.config.n_embd), n_layers=2)
    wrapper = AcousticConditionedLM(model, projector)

    # a handful of synthetic rows (equal length -- this test is about save/load,
    # not padding, which is covered by test_acoustic_alignment.py)
    rows = []
    for i in range(4):
        n = 12
        rows.append({
            "input_ids": torch.randint(1, VOCAB, (n,)),
            "attention_mask": torch.ones(n, dtype=torch.long),
            "labels": torch.randint(1, VOCAB, (n,)),
            "acoustic": torch.randn(3, DIM_AC),
            "acoustic_start": torch.tensor(2),
            "acoustic_len": torch.tensor(3),
        })

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    collate = lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]}  # noqa: E731

    class NoOp:
        pass

    args = TrainingArguments(output_dir=str(tmp / "final"), max_steps=2,
                             per_device_train_batch_size=2, save_steps=1,
                             save_strategy="steps", logging_steps=1,
                             report_to="none", remove_unused_columns=False,
                             learning_rate=1e-3)
    trainer = AcousticTrainer(model=wrapper, args=args, train_dataset=DS(),
                              data_collator=collate)
    print("1. Trainer runs with the wrapper and saves at all")
    trainer.train()
    ck = tmp / "final" / "checkpoint-2"
    check("checkpoint-2 exists", ck.is_dir(), str(ck))

    print("2. the checkpoint holds a LoRA adapter, NOT the full model state dict")
    files = sorted(p.name for p in ck.glob("*"))
    has_cfg = "adapter_config.json" in files
    has_w = any("adapter_model" in f for f in files)
    check("adapter_config.json present", has_cfg, str(files))
    check("adapter weights present", has_w)
    # the exact failure this test exists for: plain Trainer would have written
    # model.safetensors holding the entire base model
    check("no full-model model.safetensors written", "model.safetensors" not in files)
    total_mb = sum(p.stat().st_size for p in ck.rglob("*") if p.is_file()) / 1e6
    check("checkpoint is adapter-sized, not model-sized (%.2f MB)" % total_mb,
          total_mb < 50.0)

    print("3. the projector was saved beside it")
    proj_path = ck / "acoustic_projector.pt"
    check("acoustic_projector.pt present", proj_path.exists())
    saved_sd = torch.load(proj_path, map_location="cpu") if proj_path.exists() else {}

    print("4. PeftModel.from_pretrained loads the checkpoint (what P4 does)")
    try:
        fresh = tiny()
        loaded = PeftModel.from_pretrained(fresh, str(ck))
        check("PeftModel.from_pretrained succeeded", True,
              "%d LoRA params" % sum(p.numel() for p in loaded.parameters() if p.requires_grad))
    except Exception as e:                                    # noqa: BLE001
        loaded = None
        check("PeftModel.from_pretrained succeeded", False, "%s: %s" % (type(e).__name__, e))

    print("5. the projector state_dict round-trips")
    p2 = AcousticProjector(DIM_AC, 64, n_layers=2)
    if saved_sd:
        missing = p2.load_state_dict(saved_sd, strict=True)
        same = all(torch.equal(p2.state_dict()[k], saved_sd[k]) for k in saved_sd)
        check("load_state_dict(strict=True) matches what was saved", same,
              "missing=%s" % (missing,))
    else:
        check("projector state_dict loadable", False, "no file")

    print()
    print("%d/%d checkpoint round-trip checks pass" % (sum(results), len(results)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
