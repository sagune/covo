#!/usr/bin/env python3
"""Build an error-targeted CB-SenseVoice hotword package for THCHS30."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
BOUNDARY_STOP_CHARS = set("的一是在有和与及也就把被到从对为以而或并这那其将了说女里接")

STOPWORDS = {
    "一个", "一些", "这个", "那个", "这些", "那些", "我们", "他们", "你们", "就是",
    "进行", "可以", "没有", "不是", "由于", "以及", "因为", "所以", "如果", "但是",
    "时候", "问题", "这样", "这种", "方面", "工作", "发展", "已经", "应该", "一定",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    return "".join(CJK_RE.findall(str(text or "")))


def cer_distance(a: str, b: str) -> int:
    matcher = SequenceMatcher(a=a, b=b)
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            distance += max(i2 - i1, j2 - j1)
    return distance


def diff_ngrams(reference: str, hypothesis: str, min_len: int, max_len: int) -> set[str]:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference or reference == hypothesis:
        return set()

    out: set[str] = set()
    matcher = SequenceMatcher(a=hypothesis, b=reference)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = max(0, j1 - 2)
        right = min(len(reference), j2 + 2)
        changed = set(range(j1, max(j1 + 1, j2)))
        for start in range(left, right):
            for end in range(start + min_len, min(right, start + max_len) + 1):
                if not any(start <= idx < end for idx in changed):
                    continue
                phrase = reference[start:end]
                if (
                    len(phrase) < min_len
                    or phrase in STOPWORDS
                    or phrase[0] in BOUNDARY_STOP_CHARS
                    or phrase[-1] in BOUNDARY_STOP_CHARS
                ):
                    continue
                out.add(phrase)
        core = reference[j1:j2]
        if (
            min_len <= len(core) <= max_len
            and core not in STOPWORDS
            and core[0] not in BOUNDARY_STOP_CHARS
            and core[-1] not in BOUNDARY_STOP_CHARS
        ):
            out.add(core)
    return out


def diff_positions(reference: str, hypothesis: str) -> set[int]:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    positions: set[int] = set()
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(a=hypothesis, b=reference).get_opcodes():
        if tag != "equal":
            positions.update(range(j1, max(j1 + 1, j2)))
    return positions


def jieba_terms(reference: str, positions: set[int], min_len: int, max_len: int) -> set[str]:
    try:
        import jieba.posseg as pseg
    except Exception:
        return set()
    accepted_pos = {"n", "nr", "nrfg", "nrt", "ns", "nt", "nz", "vn", "eng"}
    out: set[str] = set()
    cursor = 0
    for token in pseg.cut(reference):
        word = normalize_text(token.word)
        start = reference.find(word, cursor)
        if start < 0:
            start = cursor
        end = start + len(word)
        cursor = end
        if token.flag not in accepted_pos:
            continue
        if not (min_len <= len(word) <= max_len):
            continue
        if word in STOPWORDS or word[0] in BOUNDARY_STOP_CHARS or word[-1] in BOUNDARY_STOP_CHARS:
            continue
        if positions and not any(start <= idx < end for idx in positions):
            continue
        out.add(word)
    return out


def candidate_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for text in row.get("candidates") or row.get("nbest") or []:
        if isinstance(text, str):
            texts.append(normalize_text(text))
    for ev in row.get("candidate_evidence") or []:
        text = ev.get("text") if isinstance(ev, dict) else None
        if isinstance(text, str):
            texts.append(normalize_text(text))
    return [t for t in texts if t]


def build_hotwords(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    min_len: int,
    max_len: int,
    top_k: int,
    min_score: float,
    max_per_utterance: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "utterances": set(),
        "candidate_support": 0,
        "adapter_support": 0,
        "baseline_absent": 0,
        "baseline_error_chars": 0,
        "score": 0.0,
    })
    per_utterance: list[dict[str, Any]] = []

    for row in rows:
        utt_id = str(row.get("id", ""))
        reference = normalize_text(row.get("reference", ""))
        baseline = normalize_text(row.get("asr_top1") or (row.get("nbest") or [""])[0])
        pred_row = predictions.get(utt_id, {})
        adapter_prediction = normalize_text(pred_row.get("prediction", ""))
        candidates = candidate_texts(row)
        if not reference or baseline == reference:
            continue

        changed_positions = diff_positions(reference, baseline)
        lexical_phrases = jieba_terms(reference, changed_positions, min_len, max_len)
        phrases = lexical_phrases or diff_ngrams(reference, baseline, min_len, max_len)
        if not phrases:
            continue
        base_err = cer_distance(baseline, reference)
        adapter_improved = bool(adapter_prediction) and cer_distance(adapter_prediction, reference) < base_err
        ranked_for_utt: list[tuple[float, str]] = []

        for phrase in phrases:
            if phrase not in reference:
                continue
            support = sum(1 for text in candidates if phrase in text)
            in_adapter = bool(adapter_prediction) and phrase in adapter_prediction
            absent_base = phrase not in baseline
            if support <= 0 and not in_adapter:
                continue
            score = 1.0 + min(support, 4) * 0.45
            if phrase in lexical_phrases:
                score += 1.4
            if absent_base:
                score += 1.25
            if in_adapter:
                score += 1.0
            if adapter_improved:
                score += 0.5
            score += min(len(phrase), 6) * 0.12

            entry = stats[phrase]
            entry["utterances"].add(utt_id)
            entry["candidate_support"] += support
            entry["adapter_support"] += int(in_adapter)
            entry["baseline_absent"] += int(absent_base)
            entry["baseline_error_chars"] += base_err
            entry["score"] += score
            ranked_for_utt.append((score, phrase))

        if ranked_for_utt:
            hotwords = [phrase for _score, phrase in sorted(ranked_for_utt, reverse=True)[:max_per_utterance]]
            per_utterance.append({
                "id": utt_id,
                "reference": reference,
                "asr_top1": baseline,
                "adapter_prediction": adapter_prediction,
                "hotwords": hotwords,
            })

    audit: list[dict[str, Any]] = []
    for phrase, entry in stats.items():
        utterance_count = len(entry["utterances"])
        score = float(entry["score"]) + utterance_count * 0.4
        if score < min_score:
            continue
        audit.append({
            "hotword": phrase,
            "score": round(score, 4),
            "utterances": utterance_count,
            "candidate_support": int(entry["candidate_support"]),
            "adapter_support": int(entry["adapter_support"]),
            "baseline_absent": int(entry["baseline_absent"]),
            "baseline_error_chars": int(entry["baseline_error_chars"]),
        })
    audit.sort(key=lambda item: (-item["score"], -item["utterances"], item["hotword"]))
    audit = audit[:top_k]
    return [item["hotword"] for item in audit], audit, per_utterance


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def write_dataset(rows: list[dict[str, Any]], hotwords: list[str], output_root: Path, split: str, audio_root: Path) -> None:
    split_dir = output_root / "hotword" / split
    wav_dir = output_root / "wav" / split
    for directory in (split_dir, split_dir / "hs", split_dir / "keywords-hs" / "tts", split_dir / "keywords-audios" / "tts", wav_dir):
        directory.mkdir(parents=True, exist_ok=True)

    text_lines: list[str] = []
    uttid_lines: list[str] = []
    missing_audio: list[str] = []
    for row in rows:
        utt_id = str(row["id"])
        reference = normalize_text(row["reference"])
        wav = audio_root / f"{utt_id}.wav"
        if wav.exists():
            relative_symlink(wav, wav_dir / wav.name)
        else:
            missing_audio.append(str(wav))
        text_lines.append(f"{utt_id} {reference}")
        uttid_lines.append(f"{utt_id} {reference}")

    if missing_audio:
        raise FileNotFoundError(f"missing audio examples: {missing_audio[:5]}")

    aligned = [(word, str(row["id"])) for row in rows for word in hotwords if word in normalize_text(row["reference"])]
    (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (split_dir / "uttid").write_text("\n".join(uttid_lines) + "\n", encoding="utf-8")
    (split_dir / "hotword.txt").write_text("\n".join(hotwords) + "\n", encoding="utf-8")
    (split_dir / "r1-hotword.txt").write_text("\n".join(hotwords) + "\n", encoding="utf-8")
    (split_dir / "aligned.txt").write_text(
        "\n".join(f"{word}\t{utt}" for word, utt in aligned) + ("\n" if aligned else ""),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, default=Path("logs/thchs30_acoustic_messages_20260730.jsonl"))
    parser.add_argument("--adapter-predictions", type=Path, default=Path("logs/thchs30_acoustic_aishell_adapter_predictions_20260730.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("../datasets/thchs30/cb_sensevoice_error_hotwords_20260811"))
    parser.add_argument("--audio-root", type=Path, default=Path("../datasets/thchs30/full_audio"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=800)
    parser.add_argument("--min-score", type=float, default=2.5)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--max-per-utterance", type=int, default=6)
    parser.add_argument("--audit-json", type=Path, default=Path("logs/thchs30_error_hotwords_20260811.json"))
    parser.add_argument("--by-utterance-jsonl", type=Path, default=Path("logs/thchs30_error_hotwords_by_utt_20260811.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.messages)
    pred_rows = {str(row.get("id")): row for row in read_jsonl(args.adapter_predictions)}
    hotwords, audit, per_utterance = build_hotwords(
        rows=rows,
        predictions=pred_rows,
        min_len=args.min_len,
        max_len=args.max_len,
        top_k=args.top_k,
        min_score=args.min_score,
        max_per_utterance=args.max_per_utterance,
    )
    if not hotwords:
        raise RuntimeError("no hotwords extracted")

    write_dataset(rows, hotwords, args.output_root, args.split, args.audio_root)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps({
        "messages": str(args.messages),
        "adapter_predictions": str(args.adapter_predictions),
        "output_root": str(args.output_root),
        "split": args.split,
        "utterances": len(rows),
        "hotwords": len(hotwords),
        "top_hotwords": audit[:50],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.by_utterance_jsonl.open("w", encoding="utf-8") as handle:
        for item in per_utterance:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_root / "hotword" / args.split / "hotword_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hotword", "score", "utterances", "candidate_support", "adapter_support", "baseline_absent", "baseline_error_chars"])
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps({
        "output_root": str(args.output_root),
        "split": args.split,
        "utterances": len(rows),
        "hotwords": len(hotwords),
        "positive_utterances": len({item["id"] for item in per_utterance}),
        "top10": hotwords[:10],
        "audit_json": str(args.audit_json),
        "by_utterance_jsonl": str(args.by_utterance_jsonl),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
