#!/usr/bin/env python
"""Run a base/LoRA causal LM and parse generated JSON edits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.formats import to_qwen_messages
from covo.io import read_jsonl, write_jsonl
from covo.model_output import parse_model_edits_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--input-format", choices=["internal", "qwen-messages"], default="internal")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--strict-json-system", action="store_true")
    return parser.parse_args()


def _messages_for_record(record: Dict[str, Any], input_format: str, strict_json_system: bool = False) -> List[Dict[str, str]]:
    if input_format == "qwen-messages":
        messages = list(record.get("messages", []) or [])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
    else:
        messages = to_qwen_messages(record)["messages"][:-1]
    if strict_json_system and messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": (
                "你是一个保守的中文 ASR 后纠错器。必须只输出一个合法 JSON 对象。"
                "JSON 对象只能包含 edits 字段；无需修改时输出空列表。"
                "不要输出推理过程、解释、Markdown 或示例占位符。"
            ),
        }
    return messages


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def _batched(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    if args.device != "auto":
        model.to(device)
    model.eval()

    def input_records():
        count = 0
        for record in read_jsonl(args.input):
            yield record
            count += 1
            if args.limit and count >= args.limit:
                break

    def records():
        count = 0
        batch_size = max(1, int(args.batch_size))
        for batch in _batched(input_records(), batch_size):
            prompts = []
            for record in batch:
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if args.disable_thinking:
                    template_kwargs["enable_thinking"] = False
                try:
                    prompt = tokenizer.apply_chat_template(
                        _messages_for_record(record, args.input_format, args.strict_json_system),
                        **template_kwargs,
                    )
                except TypeError:
                    template_kwargs.pop("enable_thinking", None)
                    prompt = tokenizer.apply_chat_template(
                        _messages_for_record(record, args.input_format, args.strict_json_system),
                        **template_kwargs,
                    )
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(_model_device(model))
            generation_kwargs = {
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": float(args.temperature) > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = float(args.temperature)
                generation_kwargs["top_p"] = float(args.top_p)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            prompt_width = inputs["input_ids"].shape[-1]
            for record, sequence in zip(batch, generated):
                new_tokens = sequence[prompt_width:]
                model_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                parsed = parse_model_edits_json(model_output)
                yield {
                    **record,
                    "model_output": model_output,
                    "predicted_edits": {"edits": parsed["edits"]},
                    "parse_warnings": parsed["parse_warnings"],
                }
                count += 1
            if args.progress_every and count % int(args.progress_every) == 0:
                print(json.dumps({"written": count}, ensure_ascii=False), flush=True)

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
