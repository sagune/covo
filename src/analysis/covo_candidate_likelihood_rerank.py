#!/usr/bin/env python
"""Score ASR N-best candidates with a COVO causal LM.

Unlike free-form correction, this keeps decoding inside the acoustic candidate
set.  The model score is the conditional log likelihood of the same JSON text
target used during COVO SFT; an optional CB score supplies acoustic evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

_PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    return _PUNCT_RE.sub("", text)


def unique_candidates(record: Dict[str, Any], limit: int) -> List[str]:
    input_block = record.get("input", {}) or {}
    values = [input_block.get("asr_top1", "")] + list(input_block.get("nbest", []) or [])
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
        if len(output) >= limit:
            break
    return output


def cb_scores(record: Dict[str, Any], candidates: List[str]) -> List[float]:
    scored = {}
    cb = ((record.get("input", {}) or {}).get("cbwhisper", {}) or {})
    for item in cb.get("candidates", []) or []:
        key = norm(item.get("text", ""))
        if key and key not in scored:
            scored[key] = float(item.get("total_score", 0.0) or 0.0)
    return [scored.get(norm(text), 1.0 if idx == 0 else 0.0) for idx, text in enumerate(candidates)]


def prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--cb-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else None,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    if args.device != "auto":
        model.to(device)
    model.eval()
    model_device = next(model.parameters()).device

    records = list(read_jsonl(args.input))
    if args.limit:
        records = records[: int(args.limit)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, record in enumerate(records, 1):
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if args.disable_thinking:
                template_kwargs["enable_thinking"] = False
            try:
                prompt = tokenizer.apply_chat_template(prompt_messages(record), **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                prompt = tokenizer.apply_chat_template(prompt_messages(record), **template_kwargs)

            candidates = unique_candidates(record, max(1, int(args.max_candidates)))
            acoustic = cb_scores(record, candidates)
            lm_scores: List[float] = []
            for start in range(0, len(candidates), max(1, int(args.batch_size))):
                batch = candidates[start : start + max(1, int(args.batch_size))]
                completions = [json.dumps({"text": norm(text)}, ensure_ascii=False, separators=(",", ":")) for text in batch]
                prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                encoded = tokenizer(
                    [prompt + completion for completion in completions],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                    add_special_tokens=False,
                ).to(model_device)
                with torch.inference_mode():
                    logits = model(**encoded).logits[:, :-1, :].float()
                labels = encoded.input_ids[:, 1:]
                attention = encoded.attention_mask[:, 1:].bool()
                token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                prompt_end = max(0, min(len(prompt_ids) - 1, token_logp.shape[1]))
                completion_mask = attention.clone()
                completion_mask[:, :prompt_end] = False
                sums = (token_logp * completion_mask).sum(dim=1)
                counts = completion_mask.sum(dim=1).clamp_min(1)
                lm_scores.extend((sums / counts).cpu().tolist())

            combined = [
                float(args.lm_weight) * lm + float(args.cb_weight) * cb
                for lm, cb in zip(lm_scores, acoustic)
            ]
            best_idx = max(range(len(candidates)), key=lambda idx: (combined[idx], -idx))
            scored_candidates = [
                {
                    "rank": idx + 1,
                    "text": text,
                    "lm_score": lm_scores[idx],
                    "cb_score": acoustic[idx],
                    "combined_score": combined[idx],
                }
                for idx, text in enumerate(candidates)
            ]
            result = {
                **record,
                "prediction": candidates[best_idx],
                "selected_rank": best_idx + 1,
                "candidate_scores": scored_candidates,
            }
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            if args.progress_every and row_idx % int(args.progress_every) == 0:
                print(json.dumps({"scored": row_idx}, ensure_ascii=False), flush=True)
    print(json.dumps({"input": args.input, "output": args.output, "written": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
