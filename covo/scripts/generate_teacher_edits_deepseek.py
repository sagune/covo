#!/usr/bin/env python
"""Generate teacher JSON edits with a DeepSeek/OpenAI-compatible API.

The API key is read from DEEPSEEK_API_KEY or OPENAI_API_KEY. It is never written
to output files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.formats import format_input_block
from covo.io import read_jsonl, write_jsonl
from covo.model_output import parse_model_edits_json


SYSTEM_PROMPT = (
    "你是一个保守的中文 ASR 后纠错标注器。"
    "只输出合法 JSON，格式为 {\"edits\":[{\"from\":\"...\",\"to\":\"...\",\"reason\":\"...\"}]}。"
    "只做有 N-best 或拼音证据支持的最小修改；没有必要修改时输出 {\"edits\":[]}。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("LLM_REQUEST_DELAY", "0")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LLM_TIMEOUT", "120")))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    input_block = record.get("input", {}) or record
    user = (
        "请根据以下证据输出 JSON edits。\n\n"
        f"{format_input_block(input_block)}\n\n"
        "输出要求：只输出 JSON，不要解释。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _post_chat_completion(args: argparse.Namespace, messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in the environment")
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(args.timeout)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    return str(body["choices"][0]["message"]["content"])


def main() -> int:
    args = parse_args()

    def records():
        count = 0
        for record in read_jsonl(args.input):
            messages = _build_messages(record)
            if args.dry_run:
                model_output = json.dumps({"edits": []}, ensure_ascii=False)
            else:
                model_output = _post_chat_completion(args, messages)
                if args.request_delay > 0:
                    time.sleep(float(args.request_delay))
            parsed = parse_model_edits_json(model_output)
            yield {
                **record,
                "teacher_model": args.model,
                "model_output": model_output,
                "predicted_edits": {"edits": parsed["edits"]},
                "parse_warnings": parsed["parse_warnings"],
            }
            count += 1
            if args.limit and count >= args.limit:
                break

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "model": args.model, "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
