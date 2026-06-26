#!/usr/bin/env python
"""Clean low-quality COVO n-best candidates in exported CB-Whisper evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from aishell_full_whisper_decode import edit_distance, normalize_surface


_OPENCC = None


def _cached_to_simplified(text: str) -> str:
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC

            _OPENCC = OpenCC("t2s")
        except Exception:
            _OPENCC = False
    return _OPENCC.convert(str(text)) if _OPENCC else str(text)


import aishell_full_whisper_decode as whisper_decode

whisper_decode._to_simplified = _cached_to_simplified  # type: ignore[attr-defined]


def norm(text: Any) -> str:
    return normalize_surface(str(text or ""))


def char_distance(left: Any, right: Any) -> int:
    return int(edit_distance(norm(left), norm(right)))


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def repetition_ratio(text: str) -> float:
    key = norm(text)
    if not key:
        return 1.0
    counts = Counter(key)
    return max(counts.values()) / max(len(key), 1)


def unique_texts(values: Iterable[Any]) -> List[Tuple[str, str]]:
    output = []
    seen = set()
    for value in values:
        raw = str(value or "").strip()
        key = norm(raw)
        if raw and key and key not in seen:
            output.append((raw, key))
            seen.add(key)
    return output


def clean_nbest(
    candidates: List[Any],
    max_nbest: int,
    min_ratio: float,
    max_ratio: float,
    length_slack: int,
    anchor_suffix_filter: bool = False,
    drop_polluted_top1: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    unique = unique_texts(candidates)
    if len(unique) <= 1:
        return [item[0] for item in unique[:max_nbest]], []

    pollution_markers = (
        "请不吝点赞", "订阅", "转发", "打赏", "明镜", "点点栏目",
        "优优独播", "YoYo", "Television", "Exclusive", "Series",
    )

    def severe_reason(raw: str, key: str) -> str:
        if any(marker in raw for marker in pollution_markers):
            return "bad_phrase"
        if "�" in raw:
            return "replacement_char"
        latin_alpha = sum(1 for ch in raw if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        if latin_alpha >= 8 and latin_alpha / max(len(raw), 1) > 0.20:
            return "latin_tail"
        if repetition_ratio(raw) > 0.45 and len(key) >= 8:
            return "repeat_heavy"
        return ""

    dropped: List[Dict[str, Any]] = []
    severe_clean: List[Tuple[str, str, int]] = []
    for rank, (raw, key) in enumerate(unique, start=1):
        reason = severe_reason(raw, key)
        if reason and (rank > 1 or drop_polluted_top1):
            dropped.append({"rank": rank, "text": raw, "reason": reason, "norm_len": len(key)})
            continue
        severe_clean.append((raw, key, rank))
    if len(severe_clean) == 0:
        raw, key = unique[0]
        severe_clean = [(raw, key, 1)]

    lengths = sorted(len(key) for _, key, _ in severe_clean)
    median_len = lengths[len(lengths) // 2]
    min_len = max(2, int(median_len * min_ratio))
    max_len = max(int(median_len * max_ratio), median_len + max(0, length_slack))
    shortest_key = min((key for _, key, _ in severe_clean if key), key=len, default="")
    use_short_anchor = (
        bool(anchor_suffix_filter)
        and len(shortest_key) >= 8
        and len(shortest_key) >= int(max(1, median_len) * 0.55)
    )

    kept = []
    for raw, key, rank in severe_clean:
        reason = ""
        key_len = len(key)
        if key_len < min_len:
            reason = "too_short"
        elif key_len > max_len:
            reason = "too_long"
        elif (
            use_short_anchor
            and key != shortest_key
            and key.startswith(shortest_key)
            and key_len - len(shortest_key) >= max(5, length_slack // 2)
        ):
            reason = "short_anchor_long_suffix"
        elif repetition_ratio(raw) > 0.45 and key_len >= 8:
            reason = "repeat_heavy"
        else:
            latin_alpha = sum(1 for ch in raw if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
            if latin_alpha >= 8 and latin_alpha / max(len(raw), 1) > 0.25:
                reason = "latin_tail"

        if reason:
            dropped.append({"rank": rank, "text": raw, "reason": reason, "norm_len": key_len})
            continue
        kept.append(raw)
        if len(kept) >= max_nbest:
            break
    if len(kept) == 0:
        kept = [unique[0][0]]

    return kept, dropped


def keyword_mentions(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            text = str(item.get("mention", "")).strip()
        else:
            text = str(item or "").strip()
        if text and norm(text):
            output.append(text)
    seen = set()
    deduped = []
    for item in output:
        key = norm(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def hit_count(text: str, keywords: List[str]) -> int:
    text_norm = norm(text)
    return sum(1 for keyword in keywords if norm(keyword) and norm(keyword) in text_norm)


def summarize(rows: List[Dict[str, Any]], dropped_by_row: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    count_hist = Counter()
    dropped_reasons = Counter()
    top_ed = oracle_ed = chars = exact = mentions = top_hits = oracle_hits = 0
    examples = []
    for idx, row in enumerate(rows):
        ref = str(row.get("reference", ""))
        input_block = row.get("input", {}) or {}
        nbest = list(input_block.get("nbest", []) or [])
        count_hist[len(unique_texts(nbest))] += 1
        for item in dropped_by_row.get(idx, []):
            dropped_reasons[str(item.get("reason", ""))] += 1
            if len(examples) < 12:
                examples.append({"id": row.get("id", ""), "reference": ref, **item})
        if not nbest:
            continue
        ref_norm = norm(ref)
        chars += len(ref_norm)
        top_ed += char_distance(ref, nbest[0])
        scored = [(char_distance(ref, cand), cand) for cand in nbest]
        best_ed, best_text = min(scored, key=lambda item: item[0])
        oracle_ed += best_ed
        exact += int(best_ed == 0)
        kws = keyword_mentions(input_block)
        mentions += len(kws)
        top_hits += hit_count(nbest[0], kws)
        oracle_hits += max((hit_count(cand, kws) for cand in nbest), default=0)
    rows_n = max(len(rows), 1)
    return {
        "rows": len(rows),
        "avg_unique_nbest": sum(k * v for k, v in count_hist.items()) / rows_n,
        "candidate_count_hist": dict(sorted(count_hist.items())),
        "rows_with_10_unique": sum(v for k, v in count_hist.items() if k >= 10),
        "dropped_total": sum(dropped_reasons.values()),
        "dropped_reasons": dict(dropped_reasons),
        "exact_ref_in_nbest": exact,
        "top1_corpus_cer": top_ed / max(chars, 1),
        "oracle_corpus_cer": oracle_ed / max(chars, 1),
        "top1_hotword_recall": top_hits / max(mentions, 1),
        "oracle_hotword_recall": oracle_hits / max(mentions, 1),
        "drop_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--min-ratio", type=float, default=0.65)
    parser.add_argument("--max-ratio", type=float, default=1.35)
    parser.add_argument("--length-slack", type=int, default=8)
    parser.add_argument("--anchor-suffix-filter", action="store_true")
    parser.add_argument("--keep-polluted-top1", action="store_true")
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.input)))
    dropped_by_row: Dict[int, List[Dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        input_block = row.setdefault("input", {})
        before = list(input_block.get("nbest", []) or [])
        cleaned, dropped = clean_nbest(
            candidates=before,
            max_nbest=max(1, int(args.max_nbest)),
            min_ratio=float(args.min_ratio),
            max_ratio=float(args.max_ratio),
            length_slack=int(args.length_slack),
            anchor_suffix_filter=bool(args.anchor_suffix_filter),
            drop_polluted_top1=not bool(args.keep_polluted_top1),
        )
        input_block["nbest"] = cleaned
        input_block["nbest_quality_filter"] = {
            "enabled": True,
            "before": len(unique_texts(before)),
            "after": len(cleaned),
            "dropped": dropped,
            "min_ratio": float(args.min_ratio),
            "max_ratio": float(args.max_ratio),
            "length_slack": int(args.length_slack),
        }
        if dropped:
            dropped_by_row[idx] = dropped

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = summarize(rows, dropped_by_row)
    summary.update({"input": str(args.input), "output": str(args.output)})
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
