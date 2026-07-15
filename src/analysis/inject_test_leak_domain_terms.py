#!/usr/bin/env python
"""Inject reference-derived domain phrases into COVO prompts for diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from analysis.evaluate_filler_normalized_cer import normalize_text


WARNING = "测试集泄露领域搭配（诊断专用，不能作为正式测试结果）"


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_terms(path: str | Path) -> List[str]:
    values = {normalize_text(line.strip(), []) for line in Path(path).read_text(encoding="utf-8").splitlines()}
    return sorted((value for value in values if value), key=lambda value: (-len(value), value))


def inject(record: Dict[str, Any], terms: List[str]) -> tuple[Dict[str, Any], List[str]]:
    reference = normalize_text(record.get("reference", ""), [])
    matched = [term for term in terms if term in reference]
    output = dict(record)
    output["test_leak_domain_terms"] = matched
    output["leakage_warning"] = "Uses reference-derived test terms; diagnostic only."
    messages = [dict(message) for message in record.get("messages", []) or []]
    for message in messages:
        if message.get("role") == "user" and matched:
            message["content"] = str(message.get("content", "")) + "\n" + WARNING + ": " + "、".join(matched)
            break
    output["messages"] = messages
    return output, matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--terms", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    terms = load_terms(args.terms)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = matched_rows = mentions = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in read_jsonl(args.input):
            enriched, matched = inject(record, terms)
            handle.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
            matched_rows += int(bool(matched))
            mentions += len(matched)
    print(json.dumps({"output": str(output_path), "terms": len(terms), "rows": rows,
                      "matched_rows": matched_rows, "term_mentions": mentions,
                      "leakage_warning": "diagnostic_only"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
