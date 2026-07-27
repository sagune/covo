#!/usr/bin/env python3
"""Build an oracle error-targeted hotword test set for CB-SenseVoice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import jieba

try:
    from analysis.evaluate_hotword_regions import align, hotword_mask, insertion_is_hotword, normalize
except ModuleNotFoundError:
    from evaluate_hotword_regions import align, hotword_mask, insertion_is_hotword, normalize


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def read_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["id"])] = row
    return result


def prediction_text(row: dict) -> str:
    return normalize(
        row.get("normalized_prediction")
        or row.get("prediction")
        or row.get("pred")
        or (row.get("nbest") or [""])[0]
    )


def read_text_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            utterance_id, reference = line.split(maxsplit=1)
            rows.append((utterance_id, normalize(reference)))
    return rows


def read_aligned(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0].strip() and fields[1].strip():
            result.setdefault(fields[1].strip(), []).append(normalize(fields[0]))
    return result


def target_phrases(reference: str, prediction: str, max_phrases: int, max_chars: int) -> list[str]:
    tokens = list(jieba.tokenize(reference))
    phrases = []
    for operation, position, _ in align(reference, prediction):
        if operation in {"C", "I"}:
            continue
        token_index = next(
            (index for index, (_, start, end) in enumerate(tokens) if start <= position < end),
            None,
        )
        if token_index is None:
            continue
        left = right = token_index
        while tokens[right][2] - tokens[left][1] < 2:
            if right + 1 < len(tokens):
                right += 1
            elif left > 0:
                left -= 1
            else:
                break
        start, end = tokens[left][1], tokens[right][2]
        phrase = reference[start:end]
        if len(phrase) > max_chars:
            start = max(0, min(position - max_chars // 2, len(reference) - max_chars))
            phrase = reference[start : start + max_chars]
        if 2 <= len(phrase) <= max_chars and phrase not in prediction and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def read_keywords(path: Path) -> list[str]:
    return [
        line.split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def hidden_state_cache(split_dirs: Iterable[Path]) -> dict[str, Path]:
    cache = {}
    for split_dir in split_dirs:
        keyword_file = split_dir / "hotword.txt"
        if not keyword_file.exists():
            continue
        keywords = read_keywords(keyword_file)
        width = len(str(len(keywords) - 1))
        for index, keyword in enumerate(keywords):
            path = split_dir / "keywords-hs" / "tts" / f"{index:0{width}d}.bin"
            if path.exists() and keyword not in cache:
                cache[keyword] = path
    return cache


def audio_cache(split_dirs: Iterable[Path]) -> dict[str, Path]:
    cache = {}
    for split_dir in split_dirs:
        keyword_file = split_dir / "hotword.txt"
        if not keyword_file.exists():
            continue
        keywords = read_keywords(keyword_file)
        width = len(str(len(keywords) - 1))
        for index, keyword in enumerate(keywords):
            if keyword in cache:
                continue
            stem = f"{index:0{width}d}"
            for extension in (".mp3", ".wav", ".opus"):
                path = split_dir / "keywords-audios" / "tts" / f"{stem}{extension}"
                if path.exists():
                    cache[keyword] = path
                    break
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reuse-hs-split", type=Path, action="append", default=[])
    parser.add_argument("--reuse-audio-split", type=Path, action="append", default=[])
    parser.add_argument("--max-targets-per-utterance", type=int, default=2)
    parser.add_argument("--max-target-chars", type=int, default=8)
    args = parser.parse_args()

    source_split = args.source_root / "hotword" / "test"
    output_split = args.output_root / "hotword" / "test"
    output_hs = output_split / "hs"
    output_keyword_hs = output_split / "keywords-hs" / "tts"
    output_keyword_audio = output_split / "keywords-audios" / "tts"
    for directory in (output_split, output_hs, output_keyword_hs, output_keyword_audio):
        directory.mkdir(parents=True, exist_ok=True)
    relative_symlink(args.source_root / "wav", args.output_root / "wav")

    rows = read_text_rows(source_split / "text")
    predictions = read_jsonl(args.predictions)
    missing_predictions = [utterance_id for utterance_id, _ in rows if utterance_id not in predictions]
    if missing_predictions:
        raise ValueError(f"missing {len(missing_predictions)} predictions, first={missing_predictions[0]}")

    existing_aligned = read_aligned(source_split / "aligned.txt")
    targeted: dict[str, list[str]] = {}
    for utterance_id, reference in rows:
        targeted[utterance_id] = target_phrases(
            reference,
            prediction_text(predictions[utterance_id]),
            max_phrases=max(1, args.max_targets_per_utterance),
            max_chars=max(2, args.max_target_chars),
        )

    base_keywords = read_keywords(source_split / "hotword.txt")
    target_keywords = [
        phrase
        for utterance_id, _ in rows
        for phrase in targeted[utterance_id]
    ]
    keywords = list(dict.fromkeys([*base_keywords, *target_keywords]))
    aligned = {
        utterance_id: list(dict.fromkeys([*existing_aligned.get(utterance_id, []), *targeted[utterance_id]]))
        for utterance_id, _ in rows
    }

    text = "\n".join(f"{utterance_id} {reference}" for utterance_id, reference in rows) + "\n"
    (output_split / "text").write_text(text, encoding="utf-8")
    (output_split / "uttid").write_text(text, encoding="utf-8")
    (output_split / "hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    (output_split / "r1-hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    (output_split / "aligned.txt").write_text(
        "\n".join(
            f"{phrase}\t{utterance_id}"
            for utterance_id, _ in rows
            for phrase in aligned[utterance_id]
        ) + "\n",
        encoding="utf-8",
    )

    for utterance_id, _ in rows:
        source = source_split / "hs" / f"{utterance_id}.bin"
        if not source.exists():
            raise FileNotFoundError(source)
        relative_symlink(source, output_hs / source.name)

    hs_cache = hidden_state_cache([source_split, *args.reuse_hs_split])
    wav_cache = audio_cache([source_split, *args.reuse_audio_split])
    width = len(str(len(keywords) - 1))
    for index, keyword in enumerate(keywords):
        stem = f"{index:0{width}d}"
        if keyword in hs_cache:
            relative_symlink(hs_cache[keyword], output_keyword_hs / f"{stem}.bin")
        if keyword in wav_cache:
            source = wav_cache[keyword]
            relative_symlink(source, output_keyword_audio / f"{stem}{source.suffix.lower()}")

    total_chars = hotword_chars = total_edits = covered_edits = 0
    for utterance_id, reference in rows:
        prediction = prediction_text(predictions[utterance_id])
        mask = hotword_mask(reference, aligned[utterance_id])
        total_chars += len(reference)
        hotword_chars += sum(mask)
        for operation, position, _ in align(reference, prediction):
            if operation == "C":
                continue
            total_edits += 1
            covered_edits += (
                insertion_is_hotword(mask, position) if operation == "I" else mask[position]
            )

    available_hs = len(list(output_keyword_hs.glob("*.bin")))
    available_audio = sum(1 for path in output_keyword_audio.iterdir() if path.is_file())
    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "predictions": str(args.predictions),
        "utterances": len(rows),
        "base_keywords": len(base_keywords),
        "target_keywords": len(set(target_keywords)),
        "total_keywords": len(keywords),
        "positive_utterances": sum(bool(aligned[utterance_id]) for utterance_id, _ in rows),
        "hotword_character_coverage": hotword_chars / max(total_chars, 1),
        "baseline_edits": total_edits,
        "covered_baseline_edits": covered_edits,
        "baseline_error_coverage": covered_edits / max(total_edits, 1),
        "available_keyword_hidden_states": available_hs,
        "missing_keyword_hidden_states": len(keywords) - available_hs,
        "available_keyword_audios": available_audio,
        "protocol": "oracle context targeted from test references and baseline errors",
    }
    (output_split / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
