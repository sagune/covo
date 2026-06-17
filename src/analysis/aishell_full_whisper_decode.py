#!/usr/bin/env python
"""Decode the complete AISHELL split with Whisper and compute CER.

Unlike the CB-Whisper dataloader, this script iterates over every wav file in
``data_aishell/wav/<split>`` instead of the hotword subset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def _to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(str(text))
    except Exception:
        try:
            import zhconv

            return zhconv.convert(str(text), "zh-cn")
        except Exception:
            return str(text)


def _cn_digit_seq_to_str(s: str):
    table = {
        "零": "0", "〇": "0", "○": "0", "洞": "0",
        "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
        "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
    }
    out = []
    for ch in s:
        if ch not in table:
            return None
        out.append(table[ch])
    return "".join(out)


def _cn_with_units_to_int(s: str):
    digit = {
        "零": 0, "〇": 0, "○": 0, "洞": 0,
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    small = {"十": 10, "百": 100, "千": 1000}
    large = {"万": 10000, "亿": 100000000}
    total = section = number = 0
    seen_digit = False
    for ch in s:
        if ch in digit:
            number = digit[ch]
            seen_digit = True
        elif ch in small:
            if number == 0:
                number = 1
            section += number * small[ch]
            number = 0
        elif ch in large:
            section += number
            if section == 0:
                section = 1
            total += section * large[ch]
            section = number = 0
        else:
            return None
    if not seen_digit:
        return None
    return total + section + number


def _normalize_cn_numeric_chunk(s: str) -> str:
    if not s:
        return s
    if any(ch in "十百千万亿" for ch in s):
        value = _cn_with_units_to_int(s)
        return str(value) if value is not None else s
    value = _cn_digit_seq_to_str(s)
    return value if value is not None else s


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _to_simplified(text)
    text = re.sub(r"％", "%", text)
    units = "美元|元|块|港元|欧元|日元|人民币|%|％|厘米|米|公斤|千克|岁|届|次|项|家|名|位|人|所|个|小时|分钟|秒"
    text = re.sub(r"百分之([零〇○一二两三四五六七八九十百千万亿]+)", lambda m: _normalize_cn_numeric_chunk(m.group(1)) + "%", text)
    text = re.sub(r"百分之(\d+)", lambda m: str(int(m.group(1))) + "%", text)
    text = re.sub(r"([零〇○一二两三四五六七八九]{2,4})(?=年)", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)
    text = re.sub(r"([零〇○一二两三四五六七八九十]{1,3})(?=[月日号岁])", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)
    text = re.sub(rf"([零〇○一二两三四五六七八九十百千万亿]+)(?=({units}))", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)

    def repl_large(m):
        try:
            base = int(m.group(1))
        except Exception:
            return m.group(0)
        return str(base * {"万": 10000, "亿": 100000000}[m.group(2)])

    text = re.sub(r"(\d+)([万亿])(?=(美元|元|块|港元|欧元|日元|人民币))", repl_large, text)
    text = re.sub(r"\d+", lambda m: str(int(m.group(0))), text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"[·•‧・]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def edit_distance(a: Iterable[str], b: Iterable[str]) -> int:
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


def read_transcripts(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


class AishellFullDataset(Dataset):
    def __init__(self, root: Path, split: str, transcript_path: Path):
        self.root = root
        self.split = split
        self.transcripts = read_transcripts(transcript_path)
        wav_root = root / "wav" / split
        wavs = sorted(wav_root.glob("*/*.wav"))
        self.items = [
            {"id": path.stem, "audio_path": str(path), "reference": self.transcripts[path.stem]}
            for path in wavs
            if path.stem in self.transcripts
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = dict(self.items[idx])
        wav, sr = torchaudio.load(item["audio_path"])
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        item["audio"] = wav[0]
        return item


def collate(batch: List[dict]):
    return {
        "id": [item["id"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
        "reference": [item["reference"] for item in batch],
        "audio": [item["audio"] for item in batch],
    }


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("id", "")))
            except Exception:
                pass
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/datasets/aishell/data_aishell")
    parser.add_argument("--split", default="test")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--whisper-ckpt", default="openai/whisper-large-v2")
    parser.add_argument("--output-jsonl", default="/root/autodl-tmp/src/logs/aishell_full_whisper_decode.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    transcript = Path(args.transcript) if args.transcript else root / "transcript" / "aishell_transcript_v0.8.txt"
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = AishellFullDataset(root=root, split=args.split, transcript_path=transcript)
    done = load_done(output) if args.resume else set()
    if done:
        dataset.items = [item for item in dataset.items if item["id"] not in done]
    if int(args.limit) > 0:
        dataset.items = dataset.items[: int(args.limit)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    processor = WhisperProcessor.from_pretrained(args.whisper_ckpt, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_ckpt,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate)
    mode = "a" if args.resume else "w"
    total_ed = total_len = 0
    total_mean = 0.0
    rows = 0
    with output.open(mode, encoding="utf-8") as f, torch.inference_mode():
        for batch in tqdm(loader, desc=f"decode {args.split}"):
            features = processor.feature_extractor(
                [audio.numpy() for audio in batch["audio"]],
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = features.input_features.to(device=device, dtype=model.dtype)
            attention_mask = getattr(features, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pred_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language="zh",
                task="transcribe",
                return_timestamps=False,
                num_beams=int(args.num_beams),
                do_sample=False,
            )
            preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            for utt_id, audio_path, ref, pred in zip(batch["id"], batch["audio_path"], batch["reference"], preds):
                norm_ref = normalize_surface(ref)
                norm_pred = normalize_surface(pred)
                dist = edit_distance(norm_ref, norm_pred)
                ref_len = max(len(norm_ref), 1)
                row = {
                    "id": utt_id,
                    "audio_path": audio_path,
                    "reference": ref,
                    "asr_top1": str(pred).strip(),
                    "norm_reference": norm_ref,
                    "norm_asr_top1": norm_pred,
                    "edit_distance": dist,
                    "ref_len": len(norm_ref),
                    "sample_cer": dist / ref_len,
                    "whisper_ckpt": args.whisper_ckpt,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                total_ed += dist
                total_len += len(norm_ref)
                total_mean += dist / ref_len
            f.flush()
    print(json.dumps({
        "decoded_now": rows,
        "skipped_existing": len(done),
        "total_split_items": len(dataset.items) + len(done),
        "mean_sample_cer_decoded_now": total_mean / max(rows, 1),
        "corpus_cer_decoded_now": total_ed / max(total_len, 1),
        "output_jsonl": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
