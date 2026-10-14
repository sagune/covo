#!/usr/bin/env python
"""Merge an auxiliary ASR view into a CB-SenseVoice candidate pool."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_OPENCC = None


def _to_simplified(text: str) -> str:
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC

            _OPENCC = OpenCC("t2s")
        except Exception:
            _OPENCC = False
    if _OPENCC:
        return _OPENCC.convert(str(text))
    return str(text)


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _to_simplified(text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"\s+", "", text)
    return text.strip()


def edit_distance(a: str, b: str) -> int:
    left = list(a)
    right = list(b)
    dp = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        prev = dp[0]
        dp[0] = i
        for j, other in enumerate(right, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + int(char != other))
            prev = old
    return dp[-1]


BAD_PHRASES = (
    "请不吝点赞",
    "订阅",
    "转发",
    "打赏",
    "明镜",
    "点点栏目",
    "YoYo",
    "Television",
    "Exclusive",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_asr_text(row: Dict[str, Any], field: str) -> str:
    if field:
        value: Any = row
        for part in field.split("."):
            if not isinstance(value, dict):
                return ""
            value = value.get(part, "")
        return str(value or "").strip()
    for key in ("prediction", "asr_top1", "text", "hypothesis"):
        text = str(row.get(key, "") or "").strip()
        if text:
            return text
    return ""


def unique_merge(
    items: Iterable[Tuple[str, Dict[str, Any]]],
    max_items: int,
    simplify_candidates: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    texts: List[str] = []
    sources: List[Dict[str, Any]] = []
    seen = set()
    added = 0
    for text, source in items:
        text = str(text or "").strip()
        original_text = text
        if simplify_candidates:
            text = _to_simplified(text).strip()
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        texts.append(text)
        source_row = {"text": text, **source}
        if simplify_candidates and original_text and original_text != text:
            source_row["original_text"] = original_text
        sources.append(source_row)
        seen.add(key)
        added += int(bool(source.get("auxiliary_asr")))
        if len(texts) >= max_items:
            break
    return texts, sources, added


def is_complete_candidate(text: str, anchor: str, min_ratio: float, length_slack: int) -> Tuple[bool, str]:
    key = normalize_text(text)
    anchor_key = normalize_text(anchor)
    if not key:
        return False, "empty"
    if any(phrase.lower() in str(text).lower() for phrase in BAD_PHRASES):
        return False, "bad_phrase"
    if len(anchor_key) <= 0:
        return True, ""
    if len(anchor_key) <= 6:
        return True, ""
    min_len = max(2, min(int(round(len(anchor_key) * float(min_ratio))), len(anchor_key) - int(length_slack)))
    if len(key) < min_len:
        return False, f"too_short:{len(key)}<{min_len}"
    if len(key) > max(len(anchor_key) + max(8, int(length_slack) * 2), int(round(len(anchor_key) * 1.65))):
        return False, "too_long"
    return True, ""


def normalized_edit_ratio(text: str, anchor: str) -> float:
    key = normalize_text(text)
    anchor_key = normalize_text(anchor)
    denom = max(len(key), len(anchor_key), 1)
    return edit_distance(key, anchor_key) / denom


def is_anchor_consistent_candidate(
    text: str,
    source: Dict[str, Any],
    anchor: str,
    max_edit_ratio: float,
    source_pattern: str,
) -> Tuple[bool, str]:
    if max_edit_ratio <= 0:
        return True, ""
    source_name = str(source.get("source", "") or "")
    if source_pattern and re.search(source_pattern, source_name) is None:
        return True, ""
    anchor_key = normalize_text(anchor)
    key = normalize_text(text)
    if len(anchor_key) <= 6 or len(key) <= 0:
        return True, ""
    ratio = normalized_edit_ratio(text, anchor)
    if ratio > max_edit_ratio:
        return False, f"far_from_anchor:{ratio:.3f}>{max_edit_ratio:.3f}"
    return True, ""


def make_pinyin(texts: List[str]) -> List[str]:
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return []
    return [" ".join(lazy_pinyin(text)) for text in texts]


def score_pool(reference: str, nbest: List[str]) -> Tuple[int, int, int, str]:
    ref = normalize_text(reference)
    if not ref or not nbest:
        return 0, 0, 0, ""
    top1_ed = edit_distance(normalize_text(nbest[0]), ref)
    scored = [(edit_distance(normalize_text(text), ref), idx, text) for idx, text in enumerate(nbest)]
    best_ed, best_idx, best_text = min(scored, key=lambda item: (item[0], item[1]))
    return top1_ed, best_ed, best_idx, best_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cb-pool", required=True)
    parser.add_argument("--asr-view", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--asr-field", default="")
    parser.add_argument("--source-name", default="auxiliary_asr")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--asr-first", action="store_true")
    parser.add_argument("--replace-top1", action="store_true")
    parser.add_argument("--filter-incomplete", action="store_true")
    parser.add_argument("--min-length-ratio", type=float, default=0.68)
    parser.add_argument("--length-slack", type=int, default=5)
    parser.add_argument("--filter-anchor-inconsistent", action="store_true")
    parser.add_argument("--max-anchor-edit-ratio", type=float, default=0.48)
    parser.add_argument("--anchor-filter-source-pattern", default="multiprompt")
    parser.add_argument("--simplify-candidates", action="store_true")
    args = parser.parse_args()

    cb_rows = read_jsonl(Path(args.cb_pool))
    asr_rows = {str(row.get("id", "")): row for row in read_jsonl(Path(args.asr_view))}

    output_rows = []
    total_top_ed = 0
    total_oracle_ed = 0
    total_chars = 0
    exact_top1 = 0
    exact_oracle = 0
    added_rows = 0
    added_candidates = 0
    matched = 0
    dropped_incomplete = 0
    dropped_reasons: Dict[str, int] = {}
    drop_examples = []

    for row in cb_rows:
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        utt_id = str(new_row.get("id", ""))
        input_block = new_row.setdefault("input", {})
        old_nbest = list(input_block.get("nbest", []) or [])
        old_sources = list(input_block.get("nbest_sources", []) or [])
        old_items = []
        for idx, text in enumerate(old_nbest):
            source = old_sources[idx] if idx < len(old_sources) and isinstance(old_sources[idx], dict) else {}
            source = {k: v for k, v in source.items() if k != "text"}
            old_items.append((text, source or {"source": "cbwhisper_nbest", "rank": idx + 1}))

        asr_row = asr_rows.get(utt_id, {})
        asr_text = pick_asr_text(asr_row, str(args.asr_field))
        asr_item = (
            asr_text,
            {
                "source": str(args.source_name),
                "priority": -1 if args.asr_first else 6,
                "auxiliary_asr": True,
                "source_file": str(args.asr_view),
            },
        )
        if asr_text:
            matched += 1
        anchor = asr_text or str(input_block.get("asr_top1", "") or (old_nbest[0] if old_nbest else ""))
        if bool(args.filter_incomplete) or bool(args.filter_anchor_inconsistent):
            filtered_old_items = []
            for text, source in old_items:
                keep = True
                reason = ""
                if bool(args.filter_incomplete):
                    keep, reason = is_complete_candidate(
                        text=text,
                        anchor=anchor,
                        min_ratio=float(args.min_length_ratio),
                        length_slack=int(args.length_slack),
                    )
                if keep and bool(args.filter_anchor_inconsistent):
                    keep, reason = is_anchor_consistent_candidate(
                        text=text,
                        source=source,
                        anchor=anchor,
                        max_edit_ratio=float(args.max_anchor_edit_ratio),
                        source_pattern=str(args.anchor_filter_source_pattern),
                    )
                if keep:
                    filtered_old_items.append((text, source))
                    continue
                dropped_incomplete += 1
                dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
                if len(drop_examples) < 20:
                    drop_examples.append({"id": utt_id, "text": text, "reason": reason, "anchor": anchor})
            old_items = filtered_old_items
        if args.asr_first:
            merged, sources, added = unique_merge(
                [asr_item] + old_items,
                int(args.max_nbest),
                simplify_candidates=bool(args.simplify_candidates),
            )
        else:
            merged, sources, added = unique_merge(
                old_items + [asr_item],
                int(args.max_nbest),
                simplify_candidates=bool(args.simplify_candidates),
            )

        if asr_text and any(normalize_text(src.get("text", "")) == normalize_text(asr_text) for src in sources):
            added_rows += int(normalize_text(asr_text) not in {normalize_text(text) for text in old_nbest})
        added_candidates += added

        input_block["nbest"] = merged
        input_block["nbest_sources"] = sources
        input_block["nbest_pinyin"] = make_pinyin(merged)
        if bool(args.replace_top1) and merged:
            input_block["asr_top1"] = merged[0]
        input_block["auxiliary_asr_view"] = {
            "source": str(args.source_name),
            "source_file": str(args.asr_view),
            "text": asr_text,
            "asr_first": bool(args.asr_first),
            "replace_top1": bool(args.replace_top1),
        }
        output_rows.append(new_row)

        ref = str(new_row.get("reference", ""))
        ref_norm = normalize_text(ref)
        top_ed, oracle_ed, _best_idx, _best_text = score_pool(ref, merged)
        if ref_norm:
            total_chars += len(ref_norm)
            total_top_ed += top_ed
            total_oracle_ed += oracle_ed
            exact_top1 += int(top_ed == 0)
            exact_oracle += int(oracle_ed == 0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "rows": len(output_rows),
        "asr_rows": len(asr_rows),
        "matched_rows": matched,
        "asr_first": bool(args.asr_first),
        "replace_top1": bool(args.replace_top1),
        "source_name": str(args.source_name),
        "added_rows": added_rows,
        "added_candidates": added_candidates,
        "filter_incomplete": bool(args.filter_incomplete),
        "dropped_incomplete": dropped_incomplete,
        "dropped_reasons": dropped_reasons,
        "drop_examples": drop_examples,
        "avg_nbest": sum(len(row.get("input", {}).get("nbest", []) or []) for row in output_rows) / max(len(output_rows), 1),
        "top1_corpus_cer": total_top_ed / max(total_chars, 1),
        "oracle_corpus_cer": total_oracle_ed / max(total_chars, 1),
        "top1_exact": exact_top1,
        "oracle_exact": exact_oracle,
        "output": str(output_path),
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
