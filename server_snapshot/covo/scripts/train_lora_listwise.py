#!/usr/bin/env python3
"""Train a COVO LoRA adapter with multi-candidate listwise supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
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
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sft-weight", type=float, default=0.2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [{"role": str(item.get("role", "")), "content": str(item.get("content", ""))} for item in messages]


def _format_prompt(tokenizer: Any, record: Dict[str, Any], disable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(_prompt_messages(record), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(_prompt_messages(record), **kwargs)


def _load_rows(
    path: str,
    tokenizer: Any,
    max_candidates: int,
    max_length: int,
    disable_thinking: bool,
) -> List[Dict[str, Any]]:
    eos = tokenizer.eos_token or ""
    output = []
    for record in read_jsonl(path):
        candidates = [str(item).strip() for item in record.get("candidates", []) or []]
        target = int(record.get("target_index", -1))
        if target < 0 or target >= len(candidates):
            continue
        candidates = candidates[: max(2, int(max_candidates))]
        if target >= len(candidates) or len(candidates) < 2:
            continue
        prompt = _format_prompt(tokenizer, record, disable_thinking)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        responses = [
            json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + eos
            for text in candidates
        ]
        response_ids = [tokenizer(value, add_special_tokens=False).input_ids for value in responses]
        max_prompt = max(1, int(max_length) - max(max(len(ids), 1) for ids in response_ids))
        prompt_ids = prompt_ids[-max_prompt:]
        output.append(
            {
                "id": str(record.get("id", "")),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "target_index": target,
            }
        )
    return output


def _batch_candidates(row: Dict[str, Any], pad_id: int, max_length: int, device: Any) -> tuple[Any, Any]:
    import torch

    sequences = []
    labels = []
    prompt_ids = list(row["prompt_ids"])
    for response_ids in row["response_ids"]:
        response_ids = list(response_ids)[: max(1, int(max_length) - len(prompt_ids))]
        sequences.append(prompt_ids + response_ids)
        labels.append([-100] * len(prompt_ids) + response_ids)
    width = max(len(item) for item in sequences)
    input_ids = torch.tensor(
        [[pad_id] * (width - len(item)) + item for item in sequences],
        dtype=torch.long,
        device=device,
    )
    label_ids = torch.tensor(
        [[-100] * (width - len(item)) + item for item in labels],
        dtype=torch.long,
        device=device,
    )
    return input_ids, label_ids


def _sequence_scores(model: Any, input_ids: Any, labels: Any) -> tuple[Any, Any]:
    import torch.nn.functional as F

    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    attention = input_ids.ne(pad_id).long()
    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[:, :-1, :]
    shifted = labels[:, 1:]
    mask = shifted.ne(-100)
    safe = shifted.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(dim=-1).clamp_min(1)
    sums = (token_logps * mask).sum(dim=-1)
    return sums / counts, -sums / counts


def _cycle_rows(rows: List[Dict[str, Any]], epochs: float, seed: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)
    whole_epochs = max(1, int(epochs))
    fraction = max(0.0, float(epochs) - int(epochs))
    for epoch in range(whole_epochs):
        order = list(rows)
        rng.shuffle(order)
        yield from order
    if fraction > 0:
        order = list(rows)
        rng.shuffle(order)
        yield from order[: max(1, int(len(order) * fraction))]


def main() -> int:
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = _load_rows(
        args.train_file,
        tokenizer,
        args.max_candidates,
        args.max_length,
        bool(args.disable_thinking),
    )
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "sample": rows[:1]}, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        raise ValueError("no listwise rows with a reference candidate were found")

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
    model.train()
    device = next(model.parameters()).device

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate))
    planned_steps = max(1, int(len(rows) * float(args.epochs)))
    if int(args.max_steps) > 0:
        planned_steps = min(planned_steps, int(args.max_steps))
    optimizer_steps = max(1, math.ceil(planned_steps / max(1, int(args.gradient_accumulation_steps))))
    warmup_steps = int(optimizer_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    temperature = max(float(args.temperature), 1e-4)
    stats = []
    optimizer.zero_grad(set_to_none=True)
    step = optimizer_step = 0
    for row in _cycle_rows(rows, args.epochs, int(args.seed)):
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break
        input_ids, labels = _batch_candidates(row, tokenizer.pad_token_id, args.max_length, device)
        candidate_scores, candidate_nll = _sequence_scores(model, input_ids, labels)
        target = torch.tensor([int(row["target_index"])], dtype=torch.long, device=device)
        listwise_loss = F.cross_entropy((candidate_scores / temperature).unsqueeze(0), target)
        sft_loss = candidate_nll[int(row["target_index"])]
        loss = listwise_loss + float(args.sft_weight) * sft_loss
        (loss / grad_accum).backward()
        step += 1
        if step % grad_accum == 0 or step >= planned_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        if step % max(1, int(args.logging_steps)) == 0:
            item = {
                "step": step,
                "optimizer_step": optimizer_step,
                "loss": float(loss.detach().cpu()),
                "listwise_loss": float(listwise_loss.detach().cpu()),
                "sft_loss": float(sft_loss.detach().cpu()),
                "target_rank": int(row["target_index"]) + 1,
            }
            stats.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        if int(args.save_steps) > 0 and step % int(args.save_steps) == 0:
            checkpoint = output_dir / f"checkpoint-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_file": args.train_file,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "rows": len(rows),
        "steps": step,
        "optimizer_steps": optimizer_step,
        "learning_rate": float(args.learning_rate),
        "temperature": float(args.temperature),
        "sft_weight": float(args.sft_weight),
        "max_candidates": int(args.max_candidates),
        "stats_tail": stats[-10:],
    }
    (output_dir / "listwise_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "saved": True, "steps": step}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
