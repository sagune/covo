#!/usr/bin/env python
"""Generate Whisper large-v3 1-best/N-best JSONL records.

Input manifest JSONL should contain at least:

{"id": "utt1", "audio": "/path/to/audio.wav", "reference": "..."}

This script is meant for a GPU machine. It keeps dependencies optional so the
base data tooling can be used without installing torch/transformers/librosa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.text import joined_pinyin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_audio(path: str):
    import librosa

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_model(model_name: str, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = _resolve_device(device)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model


def _forced_decoder_ids(processor: Any, language: str, task: str):
    try:
        return processor.get_decoder_prompt_ids(language=language, task=task)
    except Exception:
        return None


def transcribe_record(record: Dict[str, Any], processor: Any, model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    audio_path = str(record.get("audio", ""))
    if not audio_path:
        raise ValueError(f"Missing audio path for record {record.get('id', '')}")
    audio = _load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(args.device)
    if next(model.parameters()).dtype == torch.float16:
        input_features = input_features.half()

    with torch.inference_mode():
        sequences = model.generate(
            input_features=input_features,
            forced_decoder_ids=_forced_decoder_ids(processor, args.language, args.task),
            num_beams=max(args.num_beams, args.num_return_sequences),
            num_return_sequences=args.num_return_sequences,
            do_sample=False,
            return_dict_in_generate=False,
        )
    nbest: List[str] = processor.batch_decode(sequences, skip_special_tokens=True)
    nbest = [text.strip() for text in nbest if str(text).strip()]
    return {
        "id": str(record.get("id", "")),
        "audio": audio_path,
        "source": "whisper-large-v3",
        "asr_top1": nbest[0] if nbest else "",
        "nbest": nbest,
        "nbest_pinyin": [joined_pinyin(text) for text in nbest],
        "reference": str(record.get("reference", "")),
        "metadata": {k: v for k, v in record.items() if k not in {"id", "audio", "reference"}},
    }


def main() -> int:
    args = parse_args()
    args.device = _resolve_device(args.device)
    processor, model = _build_model(args.model, args.device)

    def records():
        count = 0
        for record in read_jsonl(args.manifest):
            yield transcribe_record(record, processor, model, args)
            count += 1
            if args.limit and count >= args.limit:
                break

    written = write_jsonl(args.output, records())
    print(json.dumps({"manifest": args.manifest, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
