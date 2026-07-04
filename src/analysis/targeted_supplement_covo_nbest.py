#!/usr/bin/env python
"""Supplement low-diversity CB-Whisper evidence rows with sampled Whisper candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from clean_covo_nbest_quality import char_distance, clean_nbest, norm, summarize, unique_texts


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_uttids(path: Path) -> List[str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if parts:
                rows.append(parts[0])
    return rows


def build_wav_map(root: Path) -> Dict[str, str]:
    return {path.stem: str(path) for path in root.rglob("*.wav")}


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav[0]


def collect_hotwords(input_block: Dict[str, Any], max_hotwords: int) -> List[str]:
    output = []
    seen = set()
    for key in ("prompt_hotwords", "hotwords"):
        for item in input_block.get(key, []) or []:
            text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
            if text and text not in seen:
                output.append(text)
                seen.add(text)
            if len(output) >= max_hotwords:
                return output
    return output


def _contains(text: str, keyword: str) -> bool:
    key = norm(keyword)
    return bool(key) and key in norm(text)


def _missing_context_hotwords(input_block: Dict[str, Any], max_hotwords: int) -> List[str]:
    nbest = list(input_block.get("nbest", []) or [])
    joined = "".join(str(item or "") for item in nbest)
    return [keyword for keyword in collect_hotwords(input_block, max_hotwords) if not _contains(joined, keyword)]


def _single_prompt_missing_hotword(input_block: Dict[str, Any]) -> List[str]:
    prompts = []
    for item in input_block.get("prompt_hotwords", []) or []:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if text and norm(text):
            prompts.append(text)
    if len(prompts) != 1:
        return []
    joined = "".join(str(item or "") for item in input_block.get("nbest", []) or [])
    return [] if _contains(joined, prompts[0]) else prompts


def prompt_variants(
    hotwords: List[str],
    max_prompt_chars: int,
    target_hotword: str | None = None,
) -> List[Tuple[str, str]]:
    variants = [("none", "")]
    if not hotwords:
        return variants
    top3 = hotwords[:3]
    top6 = hotwords[:6]
    texts = [
        ("paren_top6", "(" + ",".join(top6) + ")"),
        ("natural_top6", "以下词语可能出现在语音中：" + "、".join(top6)),
        ("keyword_sentence", "可能出现的专有名词包括：" + "、".join(top6)),
        ("paren_top3", "(" + ",".join(top3) + ")"),
    ]
    if target_hotword:
        texts = [
            ("target_natural", "请特别注意语音中可能出现的专有名词：" + target_hotword),
            ("target_context", "热词：" + target_hotword + "。请根据语音内容转写完整句子。"),
            ("target_paren", "(" + target_hotword + ")"),
        ] + texts
    for name, text in texts:
        variants.append((name, text[:max_prompt_chars]))
    return variants


def select_targeted_candidates(
    anchor: str,
    generated: Iterable[str],
    target_hotwords: List[str],
    max_per_hotword: int,
) -> List[str]:
    selected: List[str] = []
    seen = {norm(anchor)}
    for keyword in target_hotwords:
        hits = []
        for pred in generated:
            text = str(pred or "").strip()
            key = norm(text)
            if not text or not key or key in seen:
                continue
            if _contains(text, keyword):
                hits.append((char_distance(anchor, text), len(key), text))
        hits.sort(key=lambda item: (item[0], item[1]))
        for _, _, text in hits[: max(0, int(max_per_hotword))]:
            key = norm(text)
            if key not in seen:
                selected.append(text)
                seen.add(key)
    return selected


def decode_candidates(
    model,
    processor,
    audio: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
    max_new_tokens: int,
) -> List[str]:
    features = processor.feature_extractor(
        audio.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    kwargs = {
        "input_features": features.input_features.to(device=device, dtype=dtype),
        "attention_mask": getattr(features, "attention_mask", None),
        "language": "zh",
        "task": "transcribe",
        "return_timestamps": False,
        "do_sample": True,
        "temperature": max(float(temperature), 1e-6),
        "top_p": float(top_p),
        "num_beams": 1,
        "num_return_sequences": int(num_return_sequences),
    }
    if kwargs["attention_mask"] is not None:
        kwargs["attention_mask"] = kwargs["attention_mask"].to(device)
    if prompt_text:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
    if int(max_new_tokens) > 0:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    pred = model.generate(**kwargs)
    return processor.tokenizer.batch_decode(pred, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--uttid", default="datasets/aishell/data_aishell/hotword/test/uttid")
    parser.add_argument("--wav-root", default="datasets/aishell/data_aishell/wav/test")
    parser.add_argument("--whisper-ckpt", default="openai/whisper-large-v3")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--target-avg", type=float, default=9.8)
    parser.add_argument("--temperatures", default="0.6,0.8,1.0")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=8)
    parser.add_argument("--max-hotwords", type=int, default=8)
    parser.add_argument("--target-missing-context", action="store_true")
    parser.add_argument("--require-single-prompt-target", action="store_true")
    parser.add_argument("--max-target-hotwords", type=int, default=1)
    parser.add_argument("--target-insert-after", type=int, default=1)
    parser.add_argument("--target-max-per-hotword", type=int, default=2)
    parser.add_argument("--max-prompt-chars", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    uttids = read_uttids(Path(args.uttid))
    if len(rows) != len(uttids):
        raise ValueError(f"row count mismatch: evidence={len(rows)} uttids={len(uttids)}")
    wav_map = build_wav_map(Path(args.wav_root))
    temperatures = [float(x.strip()) for x in str(args.temperatures).split(",") if x.strip()]

    if bool(args.target_missing_context):
        low_indices = [
            idx for idx, row in enumerate(rows)
            if (
                _single_prompt_missing_hotword(row.get("input", {}) or {})
                if bool(args.require_single_prompt_target)
                else _missing_context_hotwords(row.get("input", {}) or {}, int(args.max_hotwords))
            )
        ]
    else:
        low_indices = [
            idx for idx, row in enumerate(rows)
            if len(unique_texts((row.get("input", {}) or {}).get("nbest", []) or [])) < int(args.max_nbest)
        ]
    if int(args.limit_rows) > 0:
        low_indices = low_indices[: int(args.limit_rows)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")
    processor = WhisperProcessor.from_pretrained(args.whisper_ckpt, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_ckpt, torch_dtype=dtype).to(device)
    model.eval()

    supplement_stats = []
    dropped_by_row = {}
    with torch.inference_mode():
        for idx in tqdm(low_indices, desc="supplement low-nbest"):
            row = rows[idx]
            input_block = row.setdefault("input", {})
            before = list(input_block.get("nbest", []) or [])
            current = list(before)
            before_count = len(unique_texts(current))
            utt_id = uttids[idx]
            audio_path = wav_map.get(utt_id)
            if audio_path is None:
                raise FileNotFoundError(utt_id)
            audio = load_audio(audio_path)
            hotwords = collect_hotwords(input_block, int(args.max_hotwords))
            target_hotwords = (
                _single_prompt_missing_hotword(input_block)
                if bool(args.require_single_prompt_target)
                else _missing_context_hotwords(input_block, int(args.max_hotwords))
            )[: int(args.max_target_hotwords)]
            added_raw = []
            generated_texts = []
            target = target_hotwords[0] if bool(args.target_missing_context) and target_hotwords else None
            for prompt_name, prompt_text in prompt_variants(hotwords, int(args.max_prompt_chars), target_hotword=target):
                for temp in temperatures:
                    preds = decode_candidates(
                        model=model,
                        processor=processor,
                        audio=audio,
                        device=device,
                        dtype=dtype,
                        prompt_text=prompt_text,
                        temperature=temp,
                        top_p=float(args.top_p),
                        num_return_sequences=int(args.num_return_sequences),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    generated_texts.extend(preds)
                    current.extend(preds)
                    added_raw.extend({"text": pred, "prompt": prompt_name, "temperature": temp} for pred in preds)
                    cleaned, dropped = clean_nbest(
                        current,
                        max_nbest=int(args.max_nbest),
                        min_ratio=0.65,
                        max_ratio=1.35,
                        length_slack=8,
                    )
                    if dropped:
                        dropped_by_row.setdefault(idx, []).extend(dropped)
                    current = cleaned
                    if not bool(args.target_missing_context) and len(unique_texts(current)) >= int(args.max_nbest):
                        break
                if not bool(args.target_missing_context) and len(unique_texts(current)) >= int(args.max_nbest):
                    break
            if bool(args.target_missing_context) and target_hotwords:
                anchor = str(input_block.get("asr_top1", "") or (before[0] if before else ""))
                targeted = select_targeted_candidates(
                    anchor=anchor,
                    generated=generated_texts,
                    target_hotwords=target_hotwords,
                    max_per_hotword=int(args.target_max_per_hotword),
                )
                if targeted:
                    insert_at = max(1, min(int(args.target_insert_after), len(before)))
                    reordered = list(before[:insert_at]) + targeted + list(before[insert_at:]) + list(current)
                    current, _ = clean_nbest(
                        reordered,
                        max_nbest=int(args.max_nbest),
                        min_ratio=0.65,
                        max_ratio=1.35,
                        length_slack=8,
                    )
            input_block["nbest"] = current[: int(args.max_nbest)]
            input_block["targeted_supplement"] = {
                "enabled": True,
                "target_missing_context": bool(args.target_missing_context),
                "require_single_prompt_target": bool(args.require_single_prompt_target),
                "target_hotwords": target_hotwords,
                "before": before_count,
                "after": len(unique_texts(input_block["nbest"])),
                "raw_generated": len(added_raw),
                "temperatures": temperatures,
                "num_return_sequences": int(args.num_return_sequences),
            }
            supplement_stats.append(input_block["targeted_supplement"])

    summary = summarize(rows, dropped_by_row)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "low_rows_processed": len(low_indices),
            "supplement_improved_rows": sum(1 for item in supplement_stats if item["after"] > item["before"]),
            "target_avg": float(args.target_avg),
            "temperatures": temperatures,
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
