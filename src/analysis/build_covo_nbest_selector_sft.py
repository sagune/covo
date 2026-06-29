#!/usr/bin/env python
"""Build COVO SFT data that teaches selecting the best n-best candidate."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cbwhisper_covo_bridge import SELECTOR_SYSTEM_MESSAGE, SYSTEM_MESSAGE, build_user_prompt  # noqa: E402

DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.metrics import edit_distance  # type: ignore  # noqa: E402

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


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def unique_texts(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def char_distance(left: Any, right: Any) -> int:
    return int(edit_distance(list(norm(left)), list(norm(right))))


def best_candidate(reference: str, candidates: List[str]) -> Tuple[str, int, int]:
    best_text = ""
    best_distance = 10**9
    best_rank = 0
    for rank, candidate in enumerate(candidates, 1):
        distance = char_distance(candidate, reference)
        if distance < best_distance:
            best_text = candidate
            best_distance = distance
            best_rank = rank
    return best_text, best_distance, best_rank


def make_record(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = str(row.get("reference", "")).strip()
    input_block = dict(row.get("input", {}) or {})
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    if not reference or not asr_top1:
        return None
    nbest = unique_texts([asr_top1] + list(input_block.get("nbest", []) or []))[: int(args.max_nbest_source)]
    if not nbest:
        return None
    best_text, best_distance, best_rank = best_candidate(reference, nbest)
    base_distance = char_distance(asr_top1, reference)
    if best_distance > base_distance:
        return None
    if best_rank == 1 and base_distance > 0 and args.keep_only_oracle_improves:
        return None
    if len(norm(best_text)) < int(args.min_target_chars):
        return None

    bridge_args = argparse.Namespace(
        max_nbest=args.max_nbest_prompt,
        max_pinyin=args.max_pinyin,
        include_pinyin=args.include_pinyin,
        max_hotwords=args.max_hotwords,
        max_prompt_hotwords=args.max_prompt_hotwords,
        max_candidates_with_scores=0,
        hotword_source="all",
        protect_supported_hotwords=args.protect_supported_hotwords,
        prompt_mode="selector" if args.selector_prompt else "correction",
    )
    return {
        "id": f"{row.get('split', '')}:{row.get('id', '')}:nbest_selector",
        "source": row.get("source", "cbwhisper"),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "reference": reference,
        "messages": [
            {"role": "system", "content": SELECTOR_SYSTEM_MESSAGE if args.selector_prompt else SYSTEM_MESSAGE},
            {"role": "user", "content": build_user_prompt({**row, "input": input_block}, bridge_args)},
            {
                "role": "assistant",
                "content": json.dumps({"text": norm(best_text)}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "nbest_selector": {
            "base_distance": base_distance,
            "best_distance": best_distance,
            "best_rank": best_rank,
            "best_is_exact": best_distance == 0,
            "oracle_improves_base": best_distance < base_distance,
            "unique_nbest": len(nbest),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-nbest-source", type=int, default=10)
    parser.add_argument("--max-nbest-prompt", type=int, default=6)
    parser.add_argument("--max-pinyin", type=int, default=3)
    parser.add_argument("--max-hotwords", type=int, default=8)
    parser.add_argument("--max-prompt-hotwords", type=int, default=6)
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument("--protect-supported-hotwords", action="store_true")
    parser.add_argument("--selector-prompt", action="store_true")
    parser.add_argument("--hard-repeat", type=int, default=4)
    parser.add_argument("--exact-repeat", type=int, default=2)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--max-hard-rows", type=int, default=0)
    parser.add_argument("--max-base-rows", type=int, default=6000)
    parser.add_argument("--keep-only-oracle-improves", action="store_true")
    parser.add_argument("--min-target-chars", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    hard_rows: List[Dict[str, Any]] = []
    exact_rows: List[Dict[str, Any]] = []
    base_rows: List[Dict[str, Any]] = []
    skipped = 0
    total = 0
    for row in read_jsonl(args.input):
        total += 1
        record = make_record(row, args)
        if record is None:
            skipped += 1
            continue
        info = record["nbest_selector"]
        if info["oracle_improves_base"]:
            hard_rows.append(record)
            if info["best_is_exact"]:
                exact_rows.append(record)
        else:
            base_rows.append(record)

    rng.shuffle(hard_rows)
    rng.shuffle(exact_rows)
    rng.shuffle(base_rows)
    if int(args.max_hard_rows) > 0:
        hard_rows = hard_rows[: int(args.max_hard_rows)]
    if int(args.max_base_rows) > 0:
        base_rows = base_rows[: int(args.max_base_rows)]

    output_rows: List[Dict[str, Any]] = []
    for row in hard_rows:
        for _ in range(max(1, int(args.hard_repeat))):
            output_rows.append(row)
    for row in exact_rows:
        for _ in range(max(0, int(args.exact_repeat))):
            output_rows.append(row)
    for row in base_rows:
        for _ in range(max(0, int(args.base_repeat))):
            output_rows.append(row)
    rng.shuffle(output_rows)
    written = write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "total": total,
                "skipped": skipped,
                "hard_rows": len(hard_rows),
                "exact_hard_rows": len(exact_rows),
                "base_rows": len(base_rows),
                "hard_repeat": int(args.hard_repeat),
                "exact_repeat": int(args.exact_repeat),
                "base_repeat": int(args.base_repeat),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
