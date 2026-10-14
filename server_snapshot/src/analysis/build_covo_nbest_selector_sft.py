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

from cbsensevoice_covo_bridge import (  # noqa: E402
    CONTENT_SELECTOR_SYSTEM_MESSAGE,
    SELECTOR_SYSTEM_MESSAGE,
    SYSTEM_MESSAGE,
    build_user_prompt,
)

DEFAULT_COVO_SRC = str(Path(__file__).resolve().parents[2] / "covo" / "src")
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


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("mention") or "").strip()
    return str(item or "").strip()


def reference_hotwords(input_block: Dict[str, Any], reference: str) -> List[str]:
    ref_norm = norm(reference)
    seen = set()
    output = []
    for key in ("keyword_mentions", "prompt_hotwords", "hotwords"):
        for item in input_block.get(key, []) or []:
            text = _item_text(item)
            text_norm = norm(text)
            if text_norm and text_norm in ref_norm and text_norm not in seen:
                seen.add(text_norm)
                output.append(text)
    output.sort(key=lambda value: (len(norm(value)), norm(value)), reverse=True)
    return output


def protected_hotwords_for_target(input_block: Dict[str, Any], reference: str, candidates: List[str]) -> List[str]:
    source_text = norm(" ".join(candidates[: max(1, min(3, len(candidates)))]))
    protected = []
    for text in reference_hotwords(input_block, reference):
        text_norm = norm(text)
        if text_norm and text_norm in source_text:
            protected.append(text)
    return protected


def candidate_contains_all(candidate: str, hotwords: List[str]) -> bool:
    candidate_norm = norm(candidate)
    return all(norm(word) in candidate_norm for word in hotwords if norm(word))


def protected_best_candidate(
    reference: str,
    candidates: List[str],
    protected_hotwords: List[str],
    margin: int,
) -> Tuple[str, int, int, bool]:
    best_text, best_distance, best_rank = best_candidate(reference, candidates)
    if not protected_hotwords:
        return best_text, best_distance, best_rank, False
    protected_options: List[Tuple[int, int, str]] = []
    for rank, candidate in enumerate(candidates, 1):
        if candidate_contains_all(candidate, protected_hotwords):
            protected_options.append((char_distance(candidate, reference), rank, candidate))
    if not protected_options:
        return best_text, best_distance, best_rank, False
    protected_distance, protected_rank, protected_text = min(protected_options, key=lambda item: (item[0], item[1]))
    if protected_distance <= best_distance + max(0, int(margin)):
        return protected_text, protected_distance, protected_rank, protected_text != best_text
    return best_text, best_distance, best_rank, False


def _system_message(prompt_mode: str) -> str:
    if prompt_mode == "selector_content":
        return CONTENT_SELECTOR_SYSTEM_MESSAGE
    if prompt_mode == "selector":
        return SELECTOR_SYSTEM_MESSAGE
    return SYSTEM_MESSAGE


def make_training_record(
    row: Dict[str, Any],
    input_block: Dict[str, Any],
    args: argparse.Namespace,
    target_text: str,
    target_distance: int,
    target_rank: int,
    base_distance: int,
    best_distance: int,
    best_rank: int,
    kind: str,
    protected_target_used: bool = False,
) -> Dict[str, Any] | None:
    reference = str(row.get("reference", "")).strip()
    if len(norm(target_text)) < int(args.min_target_chars):
        return None
    prompt_mode = str(args.prompt_mode)
    reference = str(row.get("reference", "")).strip()
    bridge_args = argparse.Namespace(
        max_nbest=args.max_nbest_prompt,
        max_pinyin=args.max_pinyin,
        include_pinyin=args.include_pinyin,
        max_hotwords=args.max_hotwords,
        max_prompt_hotwords=args.max_prompt_hotwords,
        max_candidates_with_scores=0,
        hotword_source="all",
        protect_supported_hotwords=args.protect_supported_hotwords,
        prompt_mode=prompt_mode,
    )
    return {
        "id": f"{row.get('split', '')}:{row.get('id', '')}:nbest_selector:{kind}",
        "source": row.get("source", "cbwhisper"),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "reference": reference,
        "messages": [
            {"role": "system", "content": _system_message(prompt_mode)},
            {"role": "user", "content": build_user_prompt({**row, "input": input_block}, bridge_args)},
            {
                "role": "assistant",
                "content": json.dumps({"text": norm(target_text)}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "nbest_selector": {
            "kind": kind,
            "base_distance": base_distance,
            "best_distance": best_distance,
            "best_rank": best_rank,
            "best_is_exact": best_distance == 0,
            "oracle_improves_base": best_distance < base_distance,
            "target_distance": target_distance,
            "target_rank": target_rank,
            "target_is_exact": target_distance == 0,
            "target_improves_base": target_distance < base_distance,
            "protected_target_used": bool(protected_target_used),
        },
    }


def make_records(row: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    reference = str(row.get("reference", "")).strip()
    input_block = dict(row.get("input", {}) or {})
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    if not reference or not asr_top1:
        return []
    nbest = unique_texts([asr_top1] + list(input_block.get("nbest", []) or []))[: int(args.max_nbest_source)]
    if not nbest:
        return []
    base_distance = char_distance(asr_top1, reference)
    plain_best_text, plain_best_distance, plain_best_rank = best_candidate(reference, nbest)
    protected_words = protected_hotwords_for_target(input_block, reference, nbest)
    best_text, best_distance, best_rank, protected_target_used = protected_best_candidate(
        reference=reference,
        candidates=nbest,
        protected_hotwords=protected_words if args.protect_target_hotwords else [],
        margin=int(args.protect_target_margin),
    )
    if best_distance > base_distance:
        best_text, best_distance, best_rank, protected_target_used = asr_top1, base_distance, 1, False
    if best_rank == 1 and base_distance > 0 and args.keep_only_oracle_improves:
        return []

    records = []
    selector_record = make_training_record(
        row=row,
        input_block=input_block,
        args=args,
        target_text=best_text,
        target_distance=best_distance,
        target_rank=best_rank,
        base_distance=base_distance,
        best_distance=plain_best_distance,
        best_rank=plain_best_rank,
        kind="protected_selector" if protected_target_used else "selector",
        protected_target_used=protected_target_used,
    )
    if selector_record is not None:
        selector_record["nbest_selector"]["unique_nbest"] = len(nbest)
        selector_record["nbest_selector"]["protected_hotwords"] = protected_words
        records.append(selector_record)

    if int(args.near_correct_distance) >= 0 and base_distance <= int(args.near_correct_distance):
        no_break_record = make_training_record(
            row=row,
            input_block=input_block,
            args=args,
            target_text=asr_top1,
            target_distance=base_distance,
            target_rank=1,
            base_distance=base_distance,
            best_distance=plain_best_distance,
            best_rank=plain_best_rank,
            kind="near_correct_no_break",
            protected_target_used=False,
        )
        if no_break_record is not None:
            no_break_record["nbest_selector"]["unique_nbest"] = len(nbest)
            no_break_record["nbest_selector"]["protected_hotwords"] = protected_words
            records.append(no_break_record)
    return records


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
    parser.add_argument("--prompt-mode", choices=["correction", "selector", "selector_content"], default="")
    parser.add_argument("--protect-target-hotwords", action="store_true")
    parser.add_argument("--protect-target-margin", type=int, default=2)
    parser.add_argument("--near-correct-distance", type=int, default=-1)
    parser.add_argument("--near-correct-repeat", type=int, default=0)
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
    if not args.prompt_mode:
        args.prompt_mode = "selector" if args.selector_prompt else "correction"
    rng = random.Random(int(args.seed))
    hard_rows: List[Dict[str, Any]] = []
    exact_rows: List[Dict[str, Any]] = []
    base_rows: List[Dict[str, Any]] = []
    no_break_rows: List[Dict[str, Any]] = []
    protected_rows: List[Dict[str, Any]] = []
    skipped = 0
    total = 0
    for row in read_jsonl(args.input):
        total += 1
        records = make_records(row, args)
        if not records:
            skipped += 1
            continue
        for record in records:
            info = record["nbest_selector"]
            if info["kind"] == "near_correct_no_break":
                no_break_rows.append(record)
            elif info["protected_target_used"]:
                protected_rows.append(record)
                if info["target_improves_base"]:
                    hard_rows.append(record)
                    if info["target_is_exact"]:
                        exact_rows.append(record)
                else:
                    base_rows.append(record)
            elif info["target_improves_base"]:
                hard_rows.append(record)
                if info["target_is_exact"]:
                    exact_rows.append(record)
            else:
                base_rows.append(record)

    rng.shuffle(hard_rows)
    rng.shuffle(exact_rows)
    rng.shuffle(base_rows)
    rng.shuffle(no_break_rows)
    rng.shuffle(protected_rows)
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
    for row in no_break_rows:
        for _ in range(max(0, int(args.near_correct_repeat))):
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
                "near_correct_rows": len(no_break_rows),
                "protected_rows": len(protected_rows),
                "prompt_mode": args.prompt_mode,
                "protect_target_hotwords": bool(args.protect_target_hotwords),
                "protect_target_margin": int(args.protect_target_margin),
                "near_correct_distance": int(args.near_correct_distance),
                "hard_repeat": int(args.hard_repeat),
                "exact_repeat": int(args.exact_repeat),
                "base_repeat": int(args.base_repeat),
                "near_correct_repeat": int(args.near_correct_repeat),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
