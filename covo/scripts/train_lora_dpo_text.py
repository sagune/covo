#!/usr/bin/env python
"""Lightweight DPO training for final-text ASR correction LoRA adapters."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=896)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--sft-weight",
        type=float,
        default=0.2,
        help="Weight for chosen-response SFT loss. This anchors output format during DPO.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("prompt_messages", []) or [])
    if not messages:
        messages = list(record.get("messages", []) or [])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
    return [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages]


def _assistant_text(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _format_prompt(tokenizer: Any, record: Dict[str, Any], disable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(_prompt_messages(record), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(_prompt_messages(record), **kwargs)


def _load_pairs(path: str, tokenizer: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    eos = tokenizer.eos_token or ""
    for record in read_jsonl(path):
        prompt = _format_prompt(tokenizer, record, bool(args.disable_thinking))
        chosen = _assistant_text(record.get("chosen", ""))
        rejected = _assistant_text(record.get("rejected", ""))
        if not chosen or not rejected or chosen == rejected:
            continue
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > int(args.max_prompt_length):
            prompt_ids = prompt_ids[-int(args.max_prompt_length) :]
        chosen_ids = tokenizer(chosen + eos, add_special_tokens=False).input_ids
        rejected_ids = tokenizer(rejected + eos, add_special_tokens=False).input_ids
        rows.append(
            {
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "pair_type": str(record.get("pair_type", "")),
            }
        )
    return rows


class PreferenceCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def _build(self, prompt_ids: List[int], response_ids: List[int]) -> Dict[str, List[int]]:
        max_resp = max(1, self.max_length - len(prompt_ids))
        response_ids = response_ids[:max_resp]
        input_ids = list(prompt_ids) + list(response_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)
        return {"input_ids": input_ids, "labels": labels}

    def _pad(self, seqs: List[List[int]], pad_value: int) -> Any:
        import torch

        max_len = max(len(x) for x in seqs)
        return torch.tensor([[pad_value] * (max_len - len(x)) + x for x in seqs], dtype=torch.long)

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        chosen = [self._build(row["prompt_ids"], row["chosen_ids"]) for row in rows]
        rejected = [self._build(row["prompt_ids"], row["rejected_ids"]) for row in rows]
        return {
            "chosen_input_ids": self._pad([x["input_ids"] for x in chosen], pad_id),
            "chosen_labels": self._pad([x["labels"] for x in chosen], -100),
            "rejected_input_ids": self._pad([x["input_ids"] for x in rejected], pad_id),
            "rejected_labels": self._pad([x["labels"] for x in rejected], -100),
        }


def _sequence_logps_and_nll(model: Any, input_ids: Any, labels: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    attention_mask = input_ids.ne(model.config.pad_token_id if model.config.pad_token_id is not None else 0).long()
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_logps = (token_logps * mask).sum(dim=-1)
    token_counts = mask.sum(dim=-1).clamp_min(1)
    sequence_nll = -sequence_logps / token_counts
    return sequence_logps, sequence_nll


def _sequence_logps(model: Any, input_ids: Any, labels: Any) -> Any:
    sequence_logps, _ = _sequence_logps_and_nll(model, input_ids, labels)
    return sequence_logps


def main() -> int:
    args = parse_args()

    import random
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = _load_pairs(args.train_file, tokenizer, args)
    random.shuffle(rows)
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "sample": rows[:1]}, ensure_ascii=False, indent=2))
        return 0

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model = PeftModel.from_pretrained(ref_model, args.adapter_path, is_trainable=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    collator = PreferenceCollator(tokenizer, int(args.max_length))
    loader = DataLoader(
        rows,
        batch_size=int(args.per_device_train_batch_size),
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.learning_rate))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(args.max_steps),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    step = 0
    accum = 0
    stats = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < int(args.max_steps):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            policy_chosen, chosen_nll = _sequence_logps_and_nll(model, batch["chosen_input_ids"], batch["chosen_labels"])
            policy_rejected = _sequence_logps(model, batch["rejected_input_ids"], batch["rejected_labels"])
            with torch.no_grad():
                ref_chosen = _sequence_logps(ref_model, batch["chosen_input_ids"], batch["chosen_labels"])
                ref_rejected = _sequence_logps(ref_model, batch["rejected_input_ids"], batch["rejected_labels"])
            logits = float(args.beta) * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            dpo_loss = -F.logsigmoid(logits).mean()
            sft_loss = chosen_nll.mean()
            total_loss = dpo_loss + float(args.sft_weight) * sft_loss
            loss = total_loss / int(args.gradient_accumulation_steps)
            loss.backward()
            accum += 1
            if accum % int(args.gradient_accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                with torch.no_grad():
                    margin = (policy_chosen - policy_rejected).mean().float().item()
                    acc = (policy_chosen > policy_rejected).float().mean().item()
                    loss_value = float(total_loss.item())
                    dpo_loss_value = float(dpo_loss.item())
                    sft_loss_value = float(sft_loss.item())
                if step % int(args.logging_steps) == 0 or step == 1:
                    row = {
                        "step": step,
                        "loss": round(loss_value, 6),
                        "dpo_loss": round(dpo_loss_value, 6),
                        "sft_loss": round(sft_loss_value, 6),
                        "policy_margin": round(margin, 6),
                        "preference_acc": round(acc, 6),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    stats.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if args.save_steps and step % int(args.save_steps) == 0:
                    model.save_pretrained(output_dir / f"checkpoint-{step}")
                    tokenizer.save_pretrained(output_dir / f"checkpoint-{step}")
                if step >= int(args.max_steps):
                    break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_file": args.train_file,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "beta": float(args.beta),
        "sft_weight": float(args.sft_weight),
        "rows": len(rows),
        "stats_tail": stats[-10:],
    }
    (output_dir / "dpo_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "saved": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
