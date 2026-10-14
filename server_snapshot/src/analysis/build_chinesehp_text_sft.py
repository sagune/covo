#!/usr/bin/env python3
"""Build full-sentence ChineseHP-style Qwen SFT records from N-best JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM = (
    "你是一个保守的中文ASR后纠错器。根据N-best文本和拼音证据，输出纠错后的完整句子。"
    "只修正证据支持的识别错误；证据不足时保持第一候选。必须只输出JSON对象，格式为{\"text\":\"完整句子\"}。"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nbest", type=int, default=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for input_path in args.input:
            with input_path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    input_block = row.get("input", {}) or {}
                    nbest_source = row.get("nbest", []) or input_block.get("nbest", [])
                    nbest = [str(item).strip() for item in nbest_source if str(item).strip()][: args.max_nbest]
                    if not nbest:
                        continue
                    pinyin_source = row.get("nbest_pinyin", []) or input_block.get("nbest_pinyin", [])
                    pinyin = [str(item).strip() for item in pinyin_source][: len(nbest)]
                    user_lines = ["N-best文本："]
                    user_lines.extend(f"{idx}. {text}" for idx, text in enumerate(nbest, 1))
                    user_lines.append("N-best拼音：")
                    user_lines.extend(f"{idx}. {text}" for idx, text in enumerate(pinyin, 1))
                    user_lines.append('请输出：{"text":"纠错后的完整句子"}')
                    reference = str(row.get("reference", "")).strip()
                    record = {
                        "id": str(row.get("id", "")),
                        "split": str(row.get("split", "")),
                        "reference": reference,
                        "asr_top1": nbest[0],
                        "input": {"asr_top1": nbest[0], "nbest": nbest, "nbest_pinyin": pinyin},
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "\n".join(user_lines)},
                            {"role": "assistant", "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":"))},
                        ],
                    }
                    writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
    print(json.dumps({"input": [str(path) for path in args.input], "output": str(args.output), "written": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
