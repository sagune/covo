#!/usr/bin/env python
"""Build multi-prompt Whisper n-best for the AISHELL hotword subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from aishell_full_whisper_decode import edit_distance, normalize_surface


def read_uttid(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) == 2:
                rows.append({"utt_id": parts[0], "reference": parts[1]})
    return rows


def load_evidence(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_wav_map(wav_root: Path) -> Dict[str, str]:
    return {path.stem: str(path) for path in wav_root.glob("*/*.wav")}


def collect_hotwords(input_block: dict, max_hotwords: int) -> List[str]:
    out = []
    seen = set()
    for key in ("prompt_hotwords", "hotwords"):
        for item in input_block.get(key, []) or []:
            text = str(item.get("text", "")).strip()
            if text and text not in seen:
                out.append(text)
                seen.add(text)
            if len(out) >= max_hotwords:
                return out
    return out


def prompt_variants(hotwords: List[str], max_prompt_chars: int) -> List[tuple[str, str]]:
    variants = [("none", "")]
    if not hotwords:
        return variants
    top3 = hotwords[:3]
    top6 = hotwords[:6]
    texts = [
        ("paren_top3", "(" + ",".join(top3) + ")"),
        ("natural_top3", "以下词语可能出现在语音中：" + "、".join(top3)),
        ("natural_top6", "以下词语可能出现在语音中：" + "、".join(top6)),
    ]
    for name, text in texts:
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars]
        variants.append((name, text))
    return variants


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav[0]


def decode_one(
    model,
    processor,
    audio: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    num_beams: int,
    num_return_sequences: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[str]:
    features = processor.feature_extractor(
        audio.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = features.input_features.to(device=device, dtype=dtype)
    attention_mask = getattr(features, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    kwargs = {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "language": "zh",
        "task": "transcribe",
        "return_timestamps": False,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": temperature > 0.0,
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0.0:
        kwargs["temperature"] = max(float(temperature), 1e-6)
        kwargs["top_p"] = float(top_p)
    if prompt_text:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
    pred_ids = model.generate(**kwargs)
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)


def cer_pair(ref: str, pred: str) -> tuple[int, int, float]:
    norm_ref = normalize_surface(ref)
    norm_pred = normalize_surface(pred)
    dist = edit_distance(norm_ref, norm_pred)
    ref_len = max(len(norm_ref), 1)
    return dist, len(norm_ref), dist / ref_len


def count_hotword_hits(text: str, mentions: Iterable[str]) -> int:
    norm_text = normalize_surface(text)
    return sum(1 for kw in mentions if normalize_surface(kw) in norm_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="src/logs/cbwhisper_covo_evidence_test_full.jsonl")
    parser.add_argument("--uttid", default="datasets/aishell/data_aishell/hotword/test/uttid")
    parser.add_argument("--wav-root", default="datasets/aishell/data_aishell/wav/test")
    parser.add_argument("--whisper-ckpt", default="openai/whisper-large-v3")
    parser.add_argument("--output-jsonl", default="src/logs/aishell_hotword_multiprompt_nbest_test808.jsonl")
    parser.add_argument("--summary-json", default="src/logs/aishell_hotword_multiprompt_nbest_test808_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--temperatures", default="0,0.4,0.6")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-hotwords", type=int, default=8)
    parser.add_argument("--max-nbest", type=int, default=20)
    parser.add_argument("--max-prompt-chars", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    evidence = load_evidence(Path(args.evidence))
    utt_rows = read_uttid(Path(args.uttid))
    if len(evidence) != len(utt_rows):
        raise ValueError(f"evidence rows {len(evidence)} != uttid rows {len(utt_rows)}")
    wav_map = build_wav_map(Path(args.wav_root))
    items = []
    for ev, utt in zip(evidence, utt_rows):
        if utt["utt_id"] not in wav_map:
            raise FileNotFoundError(utt["utt_id"])
        items.append({"evidence": ev, **utt, "audio_path": wav_map[utt["utt_id"]]})
    if args.limit > 0:
        items = items[: args.limit]

    temperatures = [float(x.strip()) for x in str(args.temperatures).split(",") if x.strip()]
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")
    processor = WhisperProcessor.from_pretrained(args.whisper_ckpt, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_ckpt, torch_dtype=dtype).to(device)
    model.eval()

    totals = {
        "rows": 0,
        "unique_nbest": 0,
        "gt1": 0,
        "full_max": 0,
        "top1_ed": 0,
        "oracle_ed": 0,
        "chars": 0,
        "top1_mean": 0.0,
        "oracle_mean": 0.0,
        "total_mentions": 0,
        "top1_hotword_hits": 0,
        "oracle_hotword_hits": 0,
    }
    with output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for item in tqdm(items, desc="multiprompt nbest"):
            input_block = item["evidence"].get("input", {})
            hotwords = collect_hotwords(input_block, int(args.max_hotwords))
            mentions = [m.get("mention", "") for m in input_block.get("keyword_mentions", []) or [] if m.get("mention")]
            audio = load_audio(item["audio_path"])
            raw_candidates = []
            candidate_sources = []
            for prompt_name, prompt_text in prompt_variants(hotwords, int(args.max_prompt_chars)):
                for temp in temperatures:
                    preds = decode_one(
                        model=model,
                        processor=processor,
                        audio=audio,
                        device=device,
                        dtype=dtype,
                        prompt_text=prompt_text,
                        num_beams=max(int(args.num_beams), int(args.num_return_sequences)),
                        num_return_sequences=int(args.num_return_sequences),
                        temperature=temp,
                        top_p=float(args.top_p),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    for pred in preds:
                        raw_candidates.append(pred)
                        candidate_sources.append({"prompt": prompt_name, "temperature": temp})
            nbest = []
            nbest_sources = []
            seen = set()
            for text, source in zip(raw_candidates, candidate_sources):
                text = str(text).strip()
                key = normalize_surface(text)
                if text and key not in seen:
                    nbest.append(text)
                    nbest_sources.append(source)
                    seen.add(key)
                if len(nbest) >= int(args.max_nbest):
                    break
            top1 = nbest[0] if nbest else ""
            top_ed, ref_chars, top_cer = cer_pair(item["reference"], top1)
            best_ed = top_ed
            best_idx = 0
            for idx, cand in enumerate(nbest):
                ed, _, _ = cer_pair(item["reference"], cand)
                if ed < best_ed:
                    best_ed = ed
                    best_idx = idx
            ref_len = max(ref_chars, 1)
            oracle_cer = best_ed / ref_len
            row = {
                "id": item["utt_id"],
                "audio_path": item["audio_path"],
                "reference": item["reference"],
                "hotwords": hotwords,
                "keyword_mentions": mentions,
                "asr_top1": top1,
                "nbest": nbest,
                "nbest_sources": nbest_sources,
                "oracle_idx": best_idx,
                "top1_cer": top_cer,
                "oracle_cer": oracle_cer,
                "unique_nbest": len(nbest),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            totals["rows"] += 1
            totals["unique_nbest"] += len(nbest)
            totals["gt1"] += int(len(nbest) > 1)
            totals["full_max"] += int(len(nbest) >= int(args.max_nbest))
            totals["top1_ed"] += top_ed
            totals["oracle_ed"] += best_ed
            totals["chars"] += ref_chars
            totals["top1_mean"] += top_cer
            totals["oracle_mean"] += oracle_cer
            totals["total_mentions"] += len(mentions)
            totals["top1_hotword_hits"] += count_hotword_hits(top1, mentions)
            if nbest:
                totals["oracle_hotword_hits"] += max(count_hotword_hits(cand, mentions) for cand in nbest)

    rows = max(totals["rows"], 1)
    chars = max(totals["chars"], 1)
    mentions = max(totals["total_mentions"], 1)
    summary = {
        "rows": totals["rows"],
        "avg_unique_nbest": totals["unique_nbest"] / rows,
        "gt1": totals["gt1"],
        "full_max": totals["full_max"],
        "top1_mean_cer": totals["top1_mean"] / rows,
        "top1_corpus_cer": totals["top1_ed"] / chars,
        "oracle_mean_cer": totals["oracle_mean"] / rows,
        "oracle_corpus_cer": totals["oracle_ed"] / chars,
        "total_mentions": totals["total_mentions"],
        "top1_hotword_recall": totals["top1_hotword_hits"] / mentions,
        "oracle_hotword_recall": totals["oracle_hotword_hits"] / mentions,
        "temperatures": temperatures,
        "num_beams": int(args.num_beams),
        "num_return_sequences": int(args.num_return_sequences),
        "max_nbest": int(args.max_nbest),
        "whisper_ckpt": args.whisper_ckpt,
        "output_jsonl": str(output),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
