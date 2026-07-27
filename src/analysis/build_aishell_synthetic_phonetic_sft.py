#!/usr/bin/env python
"""Build non-overlap AISHELL phonetic/hotword SFT data.

The source transcript has word segmentation.  We mine same-pinyin word pairs
from the unused part of AISHELL train, replace one reference word with a
confusable word, and expose the local phonetic evidence plus hotword fields.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pypinyin import Style, lazy_pinyin


COVO_SRC = str(Path(__file__).resolve().parents[2] / "covo" / "src")
if COVO_SRC not in sys.path:
    sys.path.insert(0, COVO_SRC)

from covo.text import normalize_chinese_text  # type: ignore  # noqa: E402

from cbsensevoice_covo_bridge import CONTENT_SELECTOR_SYSTEM_MESSAGE  # noqa: E402


PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")
CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
PINYIN_CACHE: Dict[str, str] = {}


def norm(text: Any) -> str:
    return normalize_chinese_text(PUNCT_RE.sub("", str(text or ""))).strip()


def pinyin(text: str) -> str:
    if text not in PINYIN_CACHE:
        PINYIN_CACHE[text] = " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))
    return PINYIN_CACHE[text]


def read_transcript(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            uttid, text = parts
            tokens = [tok for tok in text.split() if tok.strip()]
            joined = norm("".join(tokens))
            if joined:
                rows.append({"id": uttid, "tokens": tokens, "reference": joined})
    return rows


def load_excluded_ids(paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    record = json.loads(line)
                    excluded.add(str(record.get("id", "")).split("#", 1)[0])
                else:
                    excluded.add(line.split()[0].split("\t")[0])
    return excluded


def build_pinyin_vocab(rows: Iterable[Dict[str, Any]], min_len: int, max_len: int, min_freq: int) -> Dict[str, List[str]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for token in row["tokens"]:
            token = norm(token)
            if not (min_len <= len(token) <= max_len):
                continue
            if not CHINESE_RE.match(token):
                continue
            counts[token] += 1
    vocab: Dict[str, set[str]] = defaultdict(set)
    for token, count in counts.items():
        if count >= min_freq:
            vocab[pinyin(token)].add(token)
    return {key: sorted(values) for key, values in vocab.items() if len(values) >= 2}


def unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        item = norm(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_messages(reference: str, corrupted: str, hotword: str, confusable: str, nbest: List[str]) -> List[Dict[str, str]]:
    evidence = "\n".join(
        [
            "Phonetic hotword span evidence:",
            f"- protected_hotwords: {hotword}",
            f"- prompt_hotwords: {hotword}",
            f"- kws_hotwords: {hotword}",
            f"- candidate_span={confusable}",
            f"- reference_span={hotword}",
            f"- candidate_pinyin={pinyin(confusable)}",
            f"- reference_pinyin={pinyin(hotword)}",
            "- variant_keeps_hotword=true for candidates containing reference_span",
            "- variant_drops_hotword=true for candidates containing candidate_span instead of reference_span",
        ]
    )
    lines = [
        "任务：根据 ASR top-1、N-best 候选、热词和音素相近证据，输出完整中文转写。",
        "规则：如果候选中的局部片段与 protected_hotwords 音素相同或高度相近，应优先恢复热词；不要无证据改写整句。",
        '必须只输出 JSON：{"text":"纠错后的完整句子"}',
        f"ASR top-1: {corrupted}",
        "N-best candidates:",
    ]
    for idx, cand in enumerate(nbest, start=1):
        lines.append(f"{idx}. {cand}")
    lines.append(evidence)
    return [
        {"role": "system", "content": CONTENT_SELECTOR_SYSTEM_MESSAGE},
        {"role": "user", "content": "\n".join(lines)},
        {"role": "assistant", "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":"))},
    ]


def build_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    excluded_ids = load_excluded_ids(args.exclude_id_file)
    filtered = [r for r in rows if r["id"] not in excluded_ids]
    filtered = [r for r in filtered if int(args.min_chars) <= len(r["reference"]) <= int(args.max_chars)]
    vocab = build_pinyin_vocab(filtered, int(args.min_word_len), int(args.max_word_len), int(args.min_word_freq))
    rng = random.Random(int(args.seed))
    rng.shuffle(filtered)
    output: List[Dict[str, Any]] = []
    for row in filtered:
        if args.limit and len(output) >= int(args.limit):
            break
        candidates = []
        for token in row["tokens"]:
            word = norm(token)
            if not (int(args.min_word_len) <= len(word) <= int(args.max_word_len)):
                continue
            choices = [item for item in vocab.get(pinyin(word), []) if item != word]
            if choices:
                candidates.append((word, choices))
        if not candidates:
            continue
        hotword, choices = rng.choice(candidates)
        confusable = rng.choice(choices)
        reference = row["reference"]
        corrupted = reference.replace(hotword, confusable, 1)
        if corrupted == reference:
            continue
        prefix = reference[: reference.find(hotword)]
        suffix = reference[reference.find(hotword) + len(hotword) :]
        nbest = unique(
            [
                corrupted,
                reference,
                prefix + confusable + suffix,
                prefix + hotword + suffix,
                reference.replace(hotword, confusable, 1),
            ]
        )
        if len(nbest) < 2:
            continue
        output.append(
            {
                "id": f"{row['id']}#synthetic_phonetic",
                "source": "aishell_train_unused_synthetic_phonetic",
                "reference": reference,
                "input": {
                    "asr_top1": corrupted,
                    "nbest": nbest[: int(args.max_candidates)],
                    "protected_hotwords": [hotword],
                    "prompt_hotwords": [hotword],
                    "kws_hotwords": [hotword],
                    "metadata": {
                        "original_id": row["id"],
                        "hotword": hotword,
                        "confusable": confusable,
                        "hotword_pinyin": pinyin(hotword),
                    },
                },
                "messages": make_messages(reference, corrupted, hotword, confusable, nbest[: int(args.max_candidates)]),
            }
        )
    return output


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--exclude-id-file", action="append", default=[])
    parser.add_argument("--limit", type=int, default=60000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=706)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument("--max-word-len", type=int, default=5)
    parser.add_argument("--min-word-freq", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(read_transcript(args.transcript), args)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    dev_size = max(0, min(int(args.dev_size), len(rows)))
    dev = rows[:dev_size]
    train = rows[dev_size:]
    train_written = write_jsonl(args.train_output, train)
    dev_written = write_jsonl(args.dev_output, dev)
    print(
        json.dumps(
            {
                "train_output": args.train_output,
                "dev_output": args.dev_output,
                "train_written": train_written,
                "dev_written": dev_written,
                "total_written": train_written + dev_written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
