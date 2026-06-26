#!/usr/bin/env python
"""Merge CB-Whisper and auxiliary Whisper candidates into a quality n-best pool.

The script is an offline diagnostic bridge: it keeps CB-Whisper's own top
candidate first, adds complementary multi-prompt candidates, removes normalized
duplicates, applies light quality checks, and reports candidate-pool oracle
metrics. The same policy can later be moved into CB-Whisper inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import aishell_full_whisper_decode as whisper_decode


_OPENCC = None


def _cached_to_simplified(text: str) -> str:
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


whisper_decode._to_simplified = _cached_to_simplified  # type: ignore[attr-defined]
edit_distance = whisper_decode.edit_distance
normalize_surface = whisper_decode.normalize_surface


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=20000)
def _norm_cached(text: str) -> str:
    return normalize_surface(text)


def norm(text: Any) -> str:
    return _norm_cached(str(text or ""))


def char_distance(left: Any, right: Any) -> int:
    return int(edit_distance(norm(left), norm(right)))


def cer_pair(reference: str, prediction: str) -> Tuple[int, int, float]:
    ref = norm(reference)
    pred = norm(prediction)
    chars = max(len(ref), 1)
    edits = int(edit_distance(ref, pred))
    return edits, len(ref), edits / chars


def unique_append(
    pool: List[Dict[str, Any]],
    seen: set[str],
    text: Any,
    source: str,
    meta: Dict[str, Any] | None = None,
) -> bool:
    raw = str(text or "").strip()
    key = norm(raw)
    if not raw or not key or key in seen:
        return False
    item = {"text": raw, "source": source}
    if meta:
        item.update(meta)
    pool.append(item)
    seen.add(key)
    return True


def repetition_ratio(text: str) -> float:
    key = norm(text)
    if not key:
        return 1.0
    counts = Counter(key)
    return max(counts.values()) / max(len(key), 1)


def is_meaningful_candidate(text: str, anchor: str, min_len: int = 2) -> bool:
    key = norm(text)
    if len(key) < min_len:
        return False
    anchor_key = norm(anchor)
    if anchor_key:
        ratio = len(key) / max(len(anchor_key), 1)
        if ratio < 0.55 or ratio > 1.65:
            return False
    if repetition_ratio(text) > 0.45 and len(key) >= 8:
        return False
    return True


def keyword_mentions(input_block: Dict[str, Any]) -> List[str]:
    out = []
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            text = str(item.get("mention", "")).strip()
        else:
            text = str(item or "").strip()
        if text and norm(text):
            out.append(text)
    seen = set()
    uniq = []
    for text in out:
        key = norm(text)
        if key not in seen:
            seen.add(key)
            uniq.append(text)
    return uniq


def count_hits(text: str, keywords: Iterable[str]) -> int:
    text_key = norm(text)
    return sum(1 for keyword in keywords if norm(keyword) and norm(keyword) in text_key)


def build_pool(cb_row: Dict[str, Any], mp_row: Dict[str, Any], max_nbest: int) -> List[Dict[str, Any]]:
    input_block = cb_row.get("input", {}) or {}
    anchor = str(input_block.get("asr_top1", "") or "")
    if not anchor:
        nbest = input_block.get("nbest", []) or []
        anchor = str(nbest[0] if nbest else "")

    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()

    unique_append(pool, seen, anchor, "cbwhisper_top1", {"priority": 0})

    for idx, cand in enumerate(input_block.get("nbest", []) or []):
        unique_append(pool, seen, cand, "cbwhisper_nbest", {"priority": 1, "rank": idx + 1})

    for idx, cand in enumerate((input_block.get("cbwhisper", {}) or {}).get("candidates", []) or []):
        if isinstance(cand, dict):
            unique_append(
                pool,
                seen,
                cand.get("text", ""),
                "cbwhisper_scored",
                {"priority": 2, "rank": cand.get("rank", idx + 1), "total_score": cand.get("total_score")},
            )

    mp_nbest = mp_row.get("nbest", []) or []
    mp_sources = mp_row.get("nbest_sources", []) or []
    for idx, cand in enumerate(mp_nbest):
        text = str(cand or "").strip()
        if not is_meaningful_candidate(text, anchor):
            continue
        src = mp_sources[idx] if idx < len(mp_sources) and isinstance(mp_sources[idx], dict) else {}
        unique_append(
            pool,
            seen,
            text,
            "multiprompt_whisper",
            {"priority": 3, "rank": idx + 1, "prompt": src.get("prompt"), "temperature": src.get("temperature")},
        )
        if len(pool) >= max_nbest:
            break

    return pool[:max_nbest]


def summarize(rows: List[Dict[str, Any]], target_nbest: int) -> Dict[str, Any]:
    totals = {
        "rows": 0,
        "unique": 0,
        "gt1": 0,
        "gte5": 0,
        "full_max": 0,
        "chars": 0,
        "top1_ed": 0,
        "oracle_ed": 0,
        "top1_mean": 0.0,
        "oracle_mean": 0.0,
        "exact_ref_in_pool": 0,
        "total_mentions": 0,
        "top1_hotword_hits": 0,
        "oracle_hotword_hits": 0,
    }
    source_counts = Counter()
    count_hist = Counter()
    examples = []
    max_nbest = 0
    for row in rows:
        ref = str(row.get("reference", ""))
        input_block = row.get("input", {}) or {}
        nbest = input_block.get("nbest", []) or []
        sources = input_block.get("nbest_sources", []) or []
        max_nbest = max(max_nbest, len(nbest))
        count_hist[str(len(nbest))] += 1
        if not nbest:
            continue
        top_ed, chars, top_cer = cer_pair(ref, nbest[0])
        scored = [(char_distance(ref, cand), idx, cand) for idx, cand in enumerate(nbest)]
        best_ed, best_idx, best_text = min(scored, key=lambda item: (item[0], item[1]))
        oracle_cer = best_ed / max(chars, 1)
        mentions = keyword_mentions(input_block)
        totals["rows"] += 1
        totals["unique"] += len(nbest)
        totals["gt1"] += int(len(nbest) > 1)
        totals["gte5"] += int(len(nbest) >= 5)
        totals["full_max"] += int(len(nbest) >= target_nbest)
        totals["chars"] += chars
        totals["top1_ed"] += top_ed
        totals["oracle_ed"] += best_ed
        totals["top1_mean"] += top_cer
        totals["oracle_mean"] += oracle_cer
        totals["exact_ref_in_pool"] += int(best_ed == 0)
        totals["total_mentions"] += len(mentions)
        totals["top1_hotword_hits"] += count_hits(nbest[0], mentions)
        totals["oracle_hotword_hits"] += max((count_hits(cand, mentions) for cand in nbest), default=0)
        if len(examples) < 5 and len(nbest) >= 8:
            examples.append(
                {
                    "id": row.get("id", ""),
                    "reference": ref,
                    "top1": nbest[0],
                    "oracle": best_text,
                    "oracle_idx": best_idx,
                    "nbest": nbest,
                    "sources": sources,
                }
            )
        for source in sources:
            if isinstance(source, dict):
                source_counts[str(source.get("source", ""))] += 1

    rows_n = max(totals["rows"], 1)
    chars = max(totals["chars"], 1)
    mentions = max(totals["total_mentions"], 1)
    return {
        "rows": totals["rows"],
        "max_nbest": max_nbest,
        "avg_unique_nbest": totals["unique"] / rows_n,
        "gt1": totals["gt1"],
        "gte5": totals["gte5"],
        "full_max": totals["full_max"],
        "exact_ref_in_pool": totals["exact_ref_in_pool"],
        "top1_mean_cer": totals["top1_mean"] / rows_n,
        "top1_corpus_cer": totals["top1_ed"] / chars,
        "oracle_mean_cer": totals["oracle_mean"] / rows_n,
        "oracle_corpus_cer": totals["oracle_ed"] / chars,
        "total_mentions": totals["total_mentions"],
        "top1_hotword_recall": totals["top1_hotword_hits"] / mentions,
        "oracle_hotword_recall": totals["oracle_hotword_hits"] / mentions,
        "source_counts": dict(source_counts),
        "candidate_count_hist": dict(sorted(count_hist.items(), key=lambda item: int(item[0]))),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cb-evidence", default="src/logs/cbwhisper_covo_evidence_test_full.jsonl")
    parser.add_argument("--multiprompt", default="src/logs/aishell_hotword_multiprompt_nbest_test808_t04.jsonl")
    parser.add_argument("--uttid", default="datasets/aishell/data_aishell/hotword/test/uttid")
    parser.add_argument("--output", default="src/logs/cbwhisper_candidate_pool_test808_t04.jsonl")
    parser.add_argument("--summary", default="src/logs/cbwhisper_candidate_pool_test808_t04_summary.json")
    parser.add_argument("--max-nbest", type=int, default=10)
    args = parser.parse_args()

    cb_rows = read_jsonl(Path(args.cb_evidence))
    mp_rows = read_jsonl(Path(args.multiprompt))
    utt_rows = []
    with Path(args.uttid).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_rows.append(parts[0])
    if not (len(cb_rows) == len(mp_rows) == len(utt_rows)):
        raise ValueError(f"row mismatch: cb={len(cb_rows)} multiprompt={len(mp_rows)} uttid={len(utt_rows)}")

    output_rows = []
    for cb_row, mp_row, utt_id in zip(cb_rows, mp_rows, utt_rows):
        row = json.loads(json.dumps(cb_row, ensure_ascii=False))
        row["id"] = utt_id
        row["source"] = "cbwhisper_candidate_pool"
        pool = build_pool(row, mp_row, int(args.max_nbest))
        input_block = row.setdefault("input", {})
        input_block["nbest"] = [item["text"] for item in pool]
        input_block["nbest_sources"] = pool
        input_block.setdefault("cbwhisper", {})["candidate_pool"] = {
            "policy": "cbwhisper_first_plus_multiprompt_quality_dedup",
            "max_nbest": int(args.max_nbest),
            "multiprompt_source": str(args.multiprompt),
            "candidate_count": len(pool),
        }
        output_rows.append(row)

    summary = summarize(output_rows, target_nbest=int(args.max_nbest))
    summary.update(
        {
            "cb_evidence": str(args.cb_evidence),
            "multiprompt": str(args.multiprompt),
            "output": str(args.output),
            "max_nbest": int(args.max_nbest),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
