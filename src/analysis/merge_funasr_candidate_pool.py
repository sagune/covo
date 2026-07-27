#!/usr/bin/env python
"""Merge FunASR transcripts into a CB-SenseVoice candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(text: Any) -> str:
    return "".join(str(text or "").split())


def unique_with_first(items: Iterable[Any], max_items: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        key = norm(text)
        if not key or key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= max_items:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cb-pool", required=True)
    parser.add_argument("--funasr", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--funasr-first", action="store_true")
    args = parser.parse_args()

    cb_rows = read_jsonl(Path(args.cb_pool))
    funasr_rows = {str(row.get("id", "")): row for row in read_jsonl(Path(args.funasr))}

    output_rows = []
    exact_funasr = 0
    added = 0
    for row in cb_rows:
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        utt_id = str(new_row.get("id", ""))
        input_block = new_row.setdefault("input", {})
        old_nbest = list(input_block.get("nbest", []) or [])
        fun_row = funasr_rows.get(utt_id, {})
        fun_text = str(fun_row.get("prediction", "")).strip()
        if fun_text:
            exact_funasr += int(int(fun_row.get("edits", 999999)) == 0)
            if args.funasr_first:
                merged = unique_with_first([fun_text] + old_nbest, int(args.max_nbest))
            else:
                merged = unique_with_first(old_nbest + [fun_text], int(args.max_nbest))
            added += int(norm(fun_text) not in {norm(item) for item in old_nbest})
            input_block["asr_top1"] = fun_text if args.funasr_first else str(input_block.get("asr_top1", ""))
            input_block["nbest"] = merged
            sources = [{"text": fun_text, "source": "funasr_sensevoice", "priority": -1}]
            for source in list(input_block.get("nbest_sources", []) or []):
                if isinstance(source, dict):
                    sources.append(source)
            input_block["nbest_sources"] = sources[: len(merged)]
            input_block["funasr_preprocess"] = {
                "model": fun_row.get("model", ""),
                "prediction": fun_text,
                "edits_vs_reference_for_diagnostic": fun_row.get("edits"),
                "source_file": str(args.funasr),
            }
        output_rows.append(new_row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "rows": len(output_rows),
        "funasr_rows": len(funasr_rows),
        "funasr_first": bool(args.funasr_first),
        "funasr_exact_diagnostic": exact_funasr,
        "funasr_added_to_pool": added,
        "avg_nbest": sum(len(row.get("input", {}).get("nbest", []) or []) for row in output_rows) / max(len(output_rows), 1),
        "output": str(output),
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
