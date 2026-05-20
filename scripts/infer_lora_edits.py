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
    return parser.parse_args()


def _messages_for_record(record: Dict[str, Any], input_format: str) -> List[Dict[str, str]]:
    if input_format == "qwen-messages":
        messages = list(record.get("messages", []) or [])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return messages
    return to_qwen_messages(record)["messages"][:-1]


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    device = _resolve_device(args.device)
    if args.device != "auto":
        model.to(device)
    model.eval()

    def records():
        count = 0
        for record in read_jsonl(args.input):
            messages = _messages_for_record(record, args.input_format)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=float(args.temperature) > 0,
                    temperature=max(float(args.temperature), 1e-6),
                    top_p=float(args.top_p),
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            model_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            parsed = parse_model_edits_json(model_output)
            yield {
                **record,
                "model_output": model_output,
                "predicted_edits": {"edits": parsed["edits"]},
                "parse_warnings": parsed["parse_warnings"],
            }
            count += 1
            if args.limit and count >= args.limit:
                break

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
