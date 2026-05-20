#!/usr/bin/env python
"""LoRA/QLoRA SFT training entry point for constrained ASR correction.

The script expects Qwen-style messages JSONL by default:

{"messages":[{"role":"system","content":"..."}, ...]}

Heavy training imports stay inside main so lightweight data tools can run
without installing GPU packages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.formats import to_qwen_messages
from covo.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-format", choices=["qwen-messages", "internal"], default="qwen-messages")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate formatting and print a sample")
    return parser.parse_args()


def _record_to_messages(record: Dict[str, Any], input_format: str) -> Dict[str, Any]:
    if input_format == "qwen-messages":
        if "messages" not in record:
            raise ValueError("qwen-messages input requires a messages field")
        return record
    if input_format == "internal":
        return to_qwen_messages(record)
    raise ValueError(f"Unsupported input format: {input_format}")


def _load_texts(path: str, tokenizer: Any, input_format: str, max_length: int) -> list[Dict[str, str]]:
    rows = []
    for record in read_jsonl(path):
        messages_record = _record_to_messages(record, input_format)
        text = tokenizer.apply_chat_template(
            messages_record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Char cap avoids accidental huge rows before tokenization.
        rows.append({"text": text[: max(1, int(max_length) * 8)]})
    return rows


def main() -> int:
    args = parse_args()

    from datasets import Dataset
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = _load_texts(args.train_file, tokenizer, args.input_format, args.max_length)
    eval_rows = _load_texts(args.eval_file, tokenizer, args.input_format, args.max_length) if args.eval_file else []
    if args.dry_run:
        print(json.dumps({"train_rows": len(train_rows), "eval_rows": len(eval_rows), "sample": train_rows[:1]}, ensure_ascii=False, indent=2))
        return 0

    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 or args.qlora else None),
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=int(args.max_length),
            truncation=True,
            padding=False,
        )

    train_dataset = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = Dataset.from_list(eval_rows).map(tokenize, batched=True, remove_columns=["text"]) if eval_rows else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
