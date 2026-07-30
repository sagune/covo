#!/usr/bin/env python
"""Run a base/LoRA causal LM and parse final-text JSON predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sort-by-prompt-length", action="store_true")
    return parser.parse_args()


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


def _batched(items: Iterable[Dict[str, Any]], batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def _extract_json_object(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("missing json object")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("unterminated json object")


def _parse_text_prediction(model_output: str, fallback: str) -> tuple[str, List[str]]:
    warnings: List[str] = []
    try:
        obj = json.loads(_extract_json_object(model_output))
    except Exception as exc:
        return fallback, [f"parse_error:{exc}"]
    if not isinstance(obj, dict):
        return fallback, ["parse_error:json_not_object"]
    value = obj.get("text", fallback)
    if not isinstance(value, str):
        warnings.append("text_not_string")
        value = fallback
    value = value.strip()
    if not value:
        warnings.append("empty_text")
        value = fallback
    return value, warnings


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output)
    completed = set()
    if args.resume and output_path.exists():
        completed = {str(record.get("id", "")) for record in read_jsonl(output_path)}

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
        records = [
            record for record in read_jsonl(args.input)
            if str(record.get("id", "")) not in completed
        ]
        if args.sort_by_prompt_length:
            records.sort(key=lambda record: sum(
                len(str(message.get("content", "")))
                for message in _prompt_messages(record)
            ))
        if args.limit:
            records = records[: int(args.limit)]
        yield from records

    def records():
        count = 0
        next_progress = int(args.progress_every)
        for batch in _batched(input_records(), max(1, int(args.batch_size))):
            prompts = []
            for record in batch:
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if args.disable_thinking:
                    template_kwargs["enable_thinking"] = False
                try:
                    prompt = tokenizer.apply_chat_template(_prompt_messages(record), **template_kwargs)
                except TypeError:
                    template_kwargs.pop("enable_thinking", None)
                    prompt = tokenizer.apply_chat_template(_prompt_messages(record), **template_kwargs)
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
                generated = model.generate(**inputs, **generation_kwargs)
            prompt_width = inputs["input_ids"].shape[-1]
            for record, sequence in zip(batch, generated):
                model_output = tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True).strip()
                fallback = str((record.get("input", {}) or {}).get("asr_top1", ""))
                prediction, parse_warnings = _parse_text_prediction(model_output, fallback=fallback)
                yield {
                    **record,
                    "model_output": model_output,
                    "raw_prediction": prediction,
                    "prediction": prediction,
                    "parse_warnings": parse_warnings,
                }
                count += 1
            if next_progress and count >= next_progress:
                print(json.dumps({"written": count}, ensure_ascii=False), flush=True)
                next_progress = (count // int(args.progress_every) + 1) * int(args.progress_every)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in records():
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            written += 1
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
