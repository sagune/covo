#!/usr/bin/env python
"""Build targeted COVO SFT rows for false-hotword rejection and n-best repair."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_COVO_SRC = str(Path(__file__).resolve().parents[2] / "covo" / "src")
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.metrics import edit_distance  # type: ignore
from covo.text import normalize_chinese_text  # type: ignore


def norm(value: Any) -> str:
    return normalize_chinese_text(str(value or ""))


def dist(left: Any, right: Any) -> int:
    return int(edit_distance(list(norm(left)), list(norm(right))))


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def prompt_false_hotwords(input_block: Dict[str, Any], reference: str) -> List[str]:
    ref_norm = norm(reference)
    rows = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if not isinstance(item, dict):
            continue
        text = norm(item.get("text", ""))
        if text and text not in ref_norm:
            rows.append(text)
    return sorted(set(rows), key=lambda item: (-len(item), item))


def nbest_oracle_distance(input_block: Dict[str, Any], reference: str) -> int:
    base = input_block.get("asr_top1", "")
    candidates = [base] + list(input_block.get("nbest", []) or [])
    distances = [dist(candidate, reference) for candidate in candidates if str(candidate or "").strip()]
    return min(distances) if distances else 10**9


def tag_record(record: Dict[str, Any]) -> Dict[str, Any] | None:
    input_block = dict(record.get("input", {}) or {})
    reference = str(record.get("reference", "")).strip()
    base = str(input_block.get("asr_top1", "")).strip()
    if not reference or not base:
        return None
    base_d = dist(base, reference)
    nbest_d = nbest_oracle_distance(input_block, reference)
    false_hotwords = prompt_false_hotwords(input_block, reference)
    false_in_base = [kw for kw in false_hotwords if kw in norm(base)]
    tags = []
    if false_in_base:
        tags.append("false_hotword_rejection")
    if nbest_d < base_d:
        tags.append("nbest_local_repair")
    if not tags:
        return None
    output = dict(record)
    output["targeted_sft_tags"] = tags
    output["targeted_sft_stats"] = {
        "base_edit_distance": base_d,
        "nbest_oracle_distance": nbest_d,
        "false_prompt_hotwords_in_base": false_in_base,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--noop-input", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-false-hotword", type=int, default=2800)
    parser.add_argument("--max-nbest-repair", type=int, default=2800)
    parser.add_argument("--max-noop", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    false_rows: List[Dict[str, Any]] = []
    nbest_rows: List[Dict[str, Any]] = []
    for record in read_jsonl(args.input):
        tagged = tag_record(record)
        if tagged is None:
            continue
        tags = set(tagged.get("targeted_sft_tags", []) or [])
        if "false_hotword_rejection" in tags:
            false_rows.append(tagged)
        elif "nbest_local_repair" in tags:
            nbest_rows.append(tagged)

    rng = random.Random(int(args.seed))
    rng.shuffle(false_rows)
    rng.shuffle(nbest_rows)
    false_rows = false_rows[: max(0, int(args.max_false_hotword))]
    nbest_rows = nbest_rows[: max(0, int(args.max_nbest_repair))]

    noop_rows: List[Dict[str, Any]] = []
    if args.noop_input and int(args.max_noop) > 0:
        noop_rows = list(read_jsonl(args.noop_input))
        rng.shuffle(noop_rows)
        noop_rows = noop_rows[: int(args.max_noop)]
        for row in noop_rows:
            row["targeted_sft_tags"] = ["noop_preservation"]

    rows = false_rows + nbest_rows + noop_rows
    rng.shuffle(rows)
    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "written": written,
                "counts": {
                    "false_hotword_rejection": len(false_rows),
                    "nbest_local_repair": len(nbest_rows),
                    "noop_preservation": len(noop_rows),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
