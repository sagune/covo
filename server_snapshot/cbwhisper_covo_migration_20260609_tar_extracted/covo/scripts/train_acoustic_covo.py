#!/usr/bin/env python
"""AC-COVO stage 2: SFT of the acoustic-conditioned COVO backend.

Same task, data, optimiser and interface as covo/scripts/train_lora_sft.py -- the
ONLY change is the acoustic channel.  That is deliberate: it makes the comparison
against the text-only run a controlled one, so any change in error-correction
behaviour is attributable to the acoustics rather than to a different objective.

The projector's last layer is zero-initialised, so step 0 is numerically the
text-only model.  `--verify-zero-init` asserts exactly that against the unwrapped
base model and exits; it is the end-to-end check that the whole acoustic plumbing
(export -> index -> span location -> injection) is wired correctly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import Trainer

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO / "src"), str(REPO / "covo" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from acoustic_conditioning import (  # noqa: E402
    AcousticConditionedLM, AcousticProjector, AcousticStore,
    build_acoustic_inputs, collate_acoustic,
)


class AcousticTrainer(Trainer):
    """Save a LoRA adapter plus the projector -- NOT the unwrapped model state dict.

    `Trainer._save` begins with

        supported_classes = (PreTrainedModel, PeftModel)
        if not isinstance(self.model, supported_classes):
            ...
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            safetensors.torch.save_file(state_dict, .../model.safetensors)

    `AcousticConditionedLM` is a plain nn.Module wrapping a PeftModel, so it matches
    NEITHER class and falls into that branch.  The consequences would have been
    silent and fatal:

      * every checkpoint would contain the ENTIRE 9B base (~18 GB) instead of a
        ~100 MB LoRA adapter, and the disk had only 32 GB free, so checkpoint 2
        would have died on a full disk;
      * no `adapter_config.json` would be written, so the
        `PeftModel.from_pretrained(base, ckpt)` that P4 performs on all nine
        checkpoints (and P5 twice) would fail on every one of them.

    This override writes what the evaluation actually loads: the adapter directory
    and `acoustic_projector.pt`.
    """

    def _save(self, output_dir=None, state_dict=None):  # noqa: D102
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        m = self.accelerator.unwrap_model(self.model)
        m.base.save_pretrained(output_dir)
        torch.save(m.projector.state_dict(),
                   os.path.join(output_dir, "acoustic_projector.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif getattr(self.data_collator, "tokenizer", None) is not None:
            self.data_collator.tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True, help="linked jsonl with messages + ac_row")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--adapter-path", default="")
    p.add_argument("--acoustic-bin", required=True)
    p.add_argument("--acoustic-index", required=True)
    p.add_argument("--acoustic-dim", type=int, default=512)
    p.add_argument("--acoustic-layers", type=int, default=2)
    p.add_argument("--acoustic-dropout", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--acoustic-lr", type=float, default=3e-4,
                   help="Separate LR for the projector. It is 19M parameters trained "
                        "FROM SCRATCH, so the fine-tuning LR that suits the pretrained "
                        "LoRA is far too small for it: with the last layer starting at "
                        "exactly zero, 2e-5 moves it by only ~0.04 over an epoch, which "
                        "would leave the channel too weak to matter and produce a false "
                        "negative. The standard recipe (LLaVA) trains the projector "
                        "orders of magnitude above the LLM's fine-tuning LR.")
    p.add_argument("--lr-scheduler-type", default="linear")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--freeze-llm", action="store_true",
                   help="train the projector only (projector-alignment ablation)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-span-fail", type=float, default=0.02)
    p.add_argument("--verify-zero-init", action="store_true")
    return p.parse_args()


class AcousticDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, store, args):
        self.rows = rows
        self.tokenizer = tokenizer
        self.store = store
        self.args = args

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return build_acoustic_inputs(
            self.tokenizer, r.get("messages") or [], int(r["ac_row"]), self.store,
            max_length=self.args.max_length,
            disable_thinking=self.args.disable_thinking, train=True)


def measure_span_fail(ds, n=500):
    """Span-location failure rate on a sample -- a wrong template shows up here."""
    n = min(n, len(ds))
    bad = 0
    for i in range(n):
        if not ds[i]["span_found"]:
            bad += 1
    return bad, n


def main() -> int:
    args = parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments)

    rows = [json.loads(l) for l in Path(args.train_file).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    store = AcousticStore(args.acoustic_bin, args.acoustic_index)
    print(json.dumps({"train_rows": len(rows), "acoustic_rows": len(store),
                      "acoustic_dim": store.dim}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None, device_map="auto")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=not args.freeze_llm)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()]))
    if args.freeze_llm:
        for p in model.parameters():
            p.requires_grad = False

    d_model = int(model.config.hidden_size)
    projector = AcousticProjector(args.acoustic_dim, d_model, args.acoustic_layers,
                                  args.acoustic_dropout).to(model.device)
    wrapper = AcousticConditionedLM(model, projector)
    n_proj = sum(p.numel() for p in projector.parameters())
    print(json.dumps({"hidden_size": d_model, "projector_params": n_proj,
                      "freeze_llm": args.freeze_llm}), flush=True)

    ds = AcousticDataset(rows, tokenizer, store, args)
    collator = lambda b: collate_acoustic(b, tokenizer.pad_token_id)  # noqa: E731
    dev = wrapper.base.device

    # ---- integration check: at init the acoustic model IS the text-only model
    if args.verify_zero_init:
        probe = collator([ds[i] for i in range(min(2, len(ds)))])
        probe = {k: v.to(dev) for k, v in probe.items()}
        wrapper.eval()
        with torch.no_grad():
            out_ac = wrapper(**probe)
            bare = {k: v for k, v in probe.items()
                    if k not in ("acoustic", "acoustic_start", "acoustic_len")}
            out_txt = wrapper.base(**bare)
        da = float((out_ac.logits.float() - out_txt.logits.float()).abs().max())
        wrapper.train()
        o = wrapper(**probe)
        o.loss.backward()
        g = projector.net[-1].weight.grad
        gnorm = float(g.norm()) if g is not None else 0.0
        bad, n = measure_span_fail(ds, 200)
        print(json.dumps({
            "verify_zero_init": True,
            "loss_acoustic": float(out_ac.loss), "loss_text_only": float(out_txt.loss),
            "max_logit_absdiff": da,
            "span_found_tokens": int(probe["acoustic_len"].sum()),
            "span_fail_sample": "%d/%d" % (bad, n),
            "projector_grad_norm": gnorm, "projector_grad_nonzero": gnorm > 0,
            "verdict": "PASS" if (da < 1e-3 and gnorm > 0) else "FAIL"}), flush=True)
        return 0 if (da < 1e-3 and gnorm > 0) else 1

    bad, n = measure_span_fail(ds, 500)
    rate = bad / max(n, 1)
    print(json.dumps({"span_fail_sample": "%d/%d" % (bad, n), "span_fail_rate": rate}), flush=True)
    if rate > args.max_span_fail:
        print(json.dumps({"FATAL": "span location failed on too many rows; the template "
                                   "assumption is wrong and acoustics would be attached to "
                                   "the wrong tokens"}), flush=True)
        return 2

    class LogOnly(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                keep = {k: v for k, v in logs.items()
                        if k in {"loss", "learning_rate", "epoch", "grad_norm"}}
                keep["step"] = int(state.global_step)
                print(json.dumps({"trainer_log": keep}), flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_strategy="steps", bf16=args.bf16,
        report_to="none", dataloader_num_workers=0, remove_unused_columns=False)

    # AcousticTrainer, not Trainer: see the class docstring -- plain Trainer would
    # write the whole 9B into every checkpoint without an adapter_config.json.
    # Two parameter groups: the projector is trained from scratch, so it gets its own
    # (much larger) learning rate; the pretrained LoRA keeps the fine-tuning LR.
    proj_ids = {id(p) for p in projector.parameters()}
    llm_params = [p for p in wrapper.parameters() if p.requires_grad and id(p) not in proj_ids]
    proj_params = [p for p in projector.parameters() if p.requires_grad]
    print(json.dumps({"param_groups": {"llm_tensors": len(llm_params),
                                       "llm_lr": args.learning_rate,
                                       "projector_tensors": len(proj_params),
                                       "projector_lr": args.acoustic_lr}}), flush=True)
    optim = torch.optim.AdamW(
        [{"params": llm_params, "lr": args.learning_rate},
         {"params": proj_params, "lr": args.acoustic_lr}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    trainer = AcousticTrainer(model=wrapper, args=targs, train_dataset=ds,
                              data_collator=collator, callbacks=[LogOnly()],
                              optimizers=(optim, None))
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    torch.save(projector.state_dict(), out / "acoustic_projector.pt")
    tokenizer.save_pretrained(str(out))
    (out / "acoustic_train_meta.json").write_text(json.dumps({
        "rows": len(rows), "acoustic_dim": args.acoustic_dim, "hidden_size": d_model,
        "projector_params": n_proj, "span_fail_rate": rate,
        "freeze_llm": args.freeze_llm, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"saved": True, "output_dir": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
