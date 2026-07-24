#!/usr/bin/env python
"""Build mixed DPO pairs for hotword preservation and conservative CER control."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def norm_text(value: Any) -> str:
    return "".join(str(value or "").split()).replace("，", "").replace(",", "").replace("。", "")


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


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


def prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [{"role": str(row.get("role", "")), "content": str(row.get("content", ""))} for row in messages]


def allowed_hotwords(input_block: Dict[str, Any], source: str) -> set[str]:
    if source == "prompt":
        rows = input_block.get("prompt_hotwords", []) or []
    elif source == "covo":
        rows = input_block.get("covo_hotwords", []) or []
    elif source == "all":
        rows = list(input_block.get("prompt_hotwords", []) or []) + list(input_block.get("hotwords", []) or [])
    else:
        raise ValueError(f"unsupported hotword source: {source}")
    return {norm_text(row.get("text", "")) for row in rows if isinstance(row, dict)}


def mentioned_hotwords(record: Dict[str, Any], source: str, min_len: int) -> List[str]:
    input_block = record.get("input", {}) or {}
    reference = norm_text(record.get("reference", ""))
    allowed = allowed_hotwords(input_block, source)
    keywords = []
    for mention in input_block.get("keyword_mentions", []) or []:
        if not isinstance(mention, dict):
            continue
        text = norm_text(mention.get("mention", ""))
        if len(text) >= min_len and text in reference and text in allowed:
            keywords.append(text)
    return sorted(set(keywords), key=lambda item: (-len(item), item))


def base_pair(record: Dict[str, Any], pair_type: str, chosen: str, rejected: str, **extra: Any) -> Dict[str, Any]:
    return {
        "id": str(record.get("id", "")),
        "source": record.get("source", "cbwhisper"),
        "split": record.get("split", ""),
        "pair_type": pair_type,
        "prompt_messages": prompt_messages(record),
        "chosen": {"text": chosen},
        "rejected": {"text": rejected},
        **extra,
    }


def hotword_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    if not reference:
        return None
    keywords = mentioned_hotwords(record, args.hotword_source, int(args.min_hotword_len))
    if not keywords:
        return None
    nbest = [norm_text(item) for item in (record.get("input", {}) or {}).get("nbest", []) or []]
    max_distance = max(int(args.min_edit_distance), int(len(reference) * float(args.max_edit_distance_ratio)))
    best_rejected = None
    best_missing: List[str] = []
    best_distance = 10**9
    for candidate in nbest[1 : max(2, int(args.max_nbest_scan))]:
        if not candidate or candidate == reference:
            continue
        missing = [keyword for keyword in keywords if keyword not in candidate]
        if not missing:
            continue
        distance = edit_distance(reference, candidate)
        if distance <= max_distance and distance < best_distance:
            best_rejected = candidate
            best_missing = missing
            best_distance = distance
    if best_rejected is None:
        return None
    return base_pair(
        record,
        "hotword_preserve_candidate_dpo",
        reference,
        best_rejected,
        missing_hotwords_in_rejected=best_missing,
        edit_distance=best_distance,
    )


def cer_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    nbest = [norm_text(item) for item in (record.get("input", {}) or {}).get("nbest", []) or []]
    nbest = [item for item in nbest[: max(2, int(args.max_nbest_scan))] if item]
    if not reference or len(nbest) < 2:
        return None
    scored = sorted(((edit_distance(reference, candidate), idx, candidate) for idx, candidate in enumerate(nbest)), key=lambda x: (x[0], x[1]))
    chosen_dist, _, chosen = scored[0]
    rejected = None
    rejected_dist = -1
    for distance, _, candidate in reversed(scored):
        if candidate != chosen and distance >= chosen_dist + int(args.cer_margin):
            rejected = candidate
            rejected_dist = distance
            break
    if rejected is None:
        return None
    return base_pair(
        record,
        "cer_candidate_dpo",
        chosen,
        rejected,
        chosen_edit_distance=chosen_dist,
        rejected_edit_distance=rejected_dist,
    )


def top1_correction_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    input_block = record.get("input", {}) or {}
    asr_top1 = norm_text(input_block.get("asr_top1", ""))
    nbest = [norm_text(item) for item in input_block.get("nbest", []) or []]
    nbest = [item for item in nbest[: max(2, int(args.max_nbest_scan))] if item]
    if not reference or not asr_top1 or len(nbest) < 2:
        return None

    top1_distance = edit_distance(reference, asr_top1)
    scored = sorted(
        (edit_distance(reference, candidate), idx, candidate)
        for idx, candidate in enumerate(nbest)
        if candidate != asr_top1
    )
    if not scored:
        return None
    chosen_distance, _, chosen = scored[0]
    if chosen_distance + int(args.top1_correction_margin) > top1_distance:
        return None
    if bool(args.top1_correction_exact_only) and chosen_distance != 0:
        return None
    return base_pair(
        record,
        "top1_correction_dpo",
        chosen,
        asr_top1,
        chosen_edit_distance=chosen_distance,
        rejected_edit_distance=top1_distance,
    )


def noop_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    input_block = record.get("input", {}) or {}
    asr_top1 = norm_text(input_block.get("asr_top1", ""))
    nbest = [norm_text(item) for item in input_block.get("nbest", []) or []]
    if not reference or not asr_top1 or edit_distance(reference, asr_top1) > int(args.noop_max_distance):
        return None
    chosen_dist = edit_distance(reference, asr_top1)
    rejected = None
    rejected_dist = -1
    for candidate in nbest[1 : max(2, int(args.max_nbest_scan))]:
        if not candidate or candidate == asr_top1:
            continue
        distance = edit_distance(reference, candidate)
        if distance >= chosen_dist + int(args.noop_margin):
            rejected = candidate
            rejected_dist = distance
            break
    if rejected is None:
        return None
    return base_pair(
        record,
        "noop_conservative_dpo",
        asr_top1,
        rejected,
        chosen_edit_distance=chosen_dist,
        rejected_edit_distance=rejected_dist,
    )


def false_hotword_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    input_block = record.get("input", {}) or {}
    asr_top1 = norm_text(input_block.get("asr_top1", ""))
    if not reference or not asr_top1:
        return None

    top1_distance = edit_distance(reference, asr_top1)
    if top1_distance > int(args.false_hotword_max_chosen_distance):
        return None
    false_hotwords = {
        keyword
        for keyword in allowed_hotwords(input_block, args.hotword_source)
        if len(keyword) >= int(args.min_hotword_len) and keyword not in reference
    }
    if not false_hotwords:
        return None

    best_rejected = None
    best_inserted: List[str] = []
    best_distance = 10**9
    for candidate in (input_block.get("nbest", []) or [])[1 : max(2, int(args.max_nbest_scan))]:
        candidate = norm_text(candidate)
        if not candidate or candidate == asr_top1:
            continue
        inserted = sorted(keyword for keyword in false_hotwords if keyword in candidate)
        if not inserted:
            continue
        distance = edit_distance(reference, candidate)
        if distance >= top1_distance + int(args.false_hotword_margin) and distance < best_distance:
            best_rejected = candidate
            best_inserted = inserted
            best_distance = distance
    if best_rejected is None:
        return None
    return base_pair(
        record,
        "false_hotword_rejection_dpo",
        asr_top1,
        best_rejected,
        false_hotwords_in_rejected=best_inserted,
        chosen_edit_distance=top1_distance,
        rejected_edit_distance=best_distance,
    )


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    hotword_rows: List[Dict[str, Any]] = []
    cer_rows: List[Dict[str, Any]] = []
    top1_correction_rows: List[Dict[str, Any]] = []
    noop_rows: List[Dict[str, Any]] = []
    false_hotword_rows: List[Dict[str, Any]] = []
    for record in read_jsonl(args.input):
        if row := hotword_pair(record, args):
            hotword_rows.append(row)
        if row := cer_pair(record, args):
            cer_rows.append(row)
        if row := top1_correction_pair(record, args):
            top1_correction_rows.append(row)
        if row := noop_pair(record, args):
            noop_rows.append(row)
        if row := false_hotword_pair(record, args):
            false_hotword_rows.append(row)

    rng = random.Random(int(args.seed))
    for rows in (hotword_rows, cer_rows, top1_correction_rows, noop_rows, false_hotword_rows):
        rng.shuffle(rows)
    if int(args.max_hotword_pairs) >= 0:
        hotword_rows = hotword_rows[: int(args.max_hotword_pairs)]
    if int(args.max_cer_pairs) >= 0:
        cer_rows = cer_rows[: int(args.max_cer_pairs)]
    if int(args.max_top1_correction_pairs) >= 0:
        top1_correction_rows = top1_correction_rows[: int(args.max_top1_correction_pairs)]
    if int(args.max_noop_pairs) >= 0:
        noop_rows = noop_rows[: int(args.max_noop_pairs)]
    if int(args.max_false_hotword_pairs) >= 0:
        false_hotword_rows = false_hotword_rows[: int(args.max_false_hotword_pairs)]
    # Keep the most specific supervision when one candidate pair satisfies
    # several objectives; repeated identical pairs would silently skew DPO.
    rows = false_hotword_rows + hotword_rows + top1_correction_rows + cer_rows + noop_rows
    unique_rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for row in rows:
        key = (
            row.get("id", ""),
            norm_text((row.get("chosen", {}) or {}).get("text", "")),
            norm_text((row.get("rejected", {}) or {}).get("text", "")),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_rows.append(row)
    rows = []
    for row in unique_rows:
        repeat = 1
        if row.get("pair_type") == "hotword_preserve_candidate_dpo":
            repeat = max(1, int(args.hotword_repeat))
        elif row.get("pair_type") == "top1_correction_dpo":
            repeat = max(1, int(args.top1_correction_repeat))
        rows.extend([row] * repeat)
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hotword-source", choices=["prompt", "covo", "all"], default="prompt")
    parser.add_argument("--min-hotword-len", type=int, default=2)
    parser.add_argument("--max-nbest-scan", type=int, default=8)
    parser.add_argument("--min-edit-distance", type=int, default=2)
    parser.add_argument("--max-edit-distance-ratio", type=float, default=0.25)
    parser.add_argument("--cer-margin", type=int, default=2)
    parser.add_argument("--top1-correction-margin", type=int, default=1)
    parser.add_argument("--top1-correction-exact-only", action="store_true")
    parser.add_argument("--noop-max-distance", type=int, default=1)
    parser.add_argument("--noop-margin", type=int, default=2)
    parser.add_argument("--false-hotword-max-chosen-distance", type=int, default=1)
    parser.add_argument("--false-hotword-margin", type=int, default=1)
    parser.add_argument("--max-hotword-pairs", type=int, default=2583)
    parser.add_argument("--max-cer-pairs", type=int, default=2583)
    parser.add_argument("--max-top1-correction-pairs", type=int, default=2583)
    parser.add_argument("--max-noop-pairs", type=int, default=2583)
    parser.add_argument("--max-false-hotword-pairs", type=int, default=2583)
    parser.add_argument("--hotword-repeat", type=int, default=1)
    parser.add_argument("--top1-correction-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["pair_type"]] = counts.get(row["pair_type"], 0) + 1
    written = write_jsonl(args.output, rows)
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
