#!/usr/bin/env python3
"""Generate ChineseHP-style SenseVoice CTC N-best hypotheses."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.sensevoice_ctc import ctc_prefix_beam_search


TAG_RE = re.compile(r"<\|[^|]+\|>")


def normalize_text(text: str) -> str:
    text = TAG_RE.sub("", str(text or "")).replace("▁", " ").strip()
    try:
        from opencc import OpenCC

        text = OpenCC("t2s").convert(text)
    except Exception:
        pass
    text = re.sub(r"\s+", "", text)
    return "".join(
        char for char in text
        if "\u4e00" <= char <= "\u9fff" or char.isdigit() or "a" <= char.lower() <= "z"
    )


def joined_pinyin(text: str) -> str:
    from pypinyin import Style, lazy_pinyin

    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors=lambda item: list(item)))


def load_completed(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line).get("id", "")))
            except (json.JSONDecodeError, AttributeError):
                continue
    return completed


def encode_audio(model, tokenizer, frontend, audio_path: str, device: torch.device, use_itn: bool):
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    audio = load_audio_text_image_video(
        audio_path,
        fs=frontend.fs,
        audio_fs=16000,
        data_type="sound",
        tokenizer=tokenizer,
    )
    speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
    speech = speech.to(device)
    speech_lengths = speech_lengths.to(device)
    language_id = model.lid_dict.get("zh", model.lid_dict["auto"])
    language_query = model.embed(torch.tensor([[language_id]], device=device)).repeat(speech.size(0), 1, 1)
    textnorm = "withitn" if use_itn else "woitn"
    textnorm_query = model.embed(torch.tensor([[model.textnorm_dict[textnorm]]], device=device)).repeat(
        speech.size(0), 1, 1
    )
    event_emo_query = model.embed(torch.tensor([[1, 2]], device=device)).repeat(speech.size(0), 1, 1)
    speech = torch.cat((language_query, event_emo_query, textnorm_query, speech), dim=1)
    speech_lengths = speech_lengths + 4
    with torch.inference_mode():
        encoder_out, encoder_out_lens = model.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        log_probs = model.ctc.log_softmax(encoder_out)
    valid_len = int(encoder_out_lens[0].item())
    return log_probs[0, min(4, valid_len):valid_len, :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--token-topk", type=int, default=24)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--use-itn", action="store_true")
    args = parser.parse_args()

    from funasr import AutoModel

    wrapper = AutoModel(model=args.model, device=args.device, disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = next(model.parameters()).device
    vocab_size = int(model.ctc.ctc_lo.out_features)
    excluded = set(range(1, 16)) | set(range(25000, vocab_size))
    completed = load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                if row_index % args.num_shards == args.shard_index and str(row.get("id", "")) not in completed:
                    rows.append(row)
    if args.limit:
        rows = rows[: args.limit]

    started = time.monotonic()
    written = 0
    with args.output.open("a", encoding="utf-8") as writer:
        for row in rows:
            log_probs = encode_audio(model, tokenizer, frontend, str(row["wav"]), device, args.use_itn)
            beams = ctc_prefix_beam_search(
                log_probs,
                beam_size=args.beam_size,
                token_topk=args.token_topk,
                blank_id=int(model.blank_id),
                excluded_token_ids=excluded,
            )
            candidates = []
            scores = []
            seen = set()
            for beam in beams:
                text = normalize_text(tokenizer.decode(list(beam.token_ids)))
                if not text or text in seen:
                    continue
                seen.add(text)
                candidates.append(text)
                scores.append(float(beam.acoustic_score))
                if len(candidates) >= args.max_nbest:
                    break
            output = {
                "dataset": args.dataset or str(row.get("source", row.get("dataset", "sensevoice"))),
                "split": str(row.get("split", "")),
                "id": str(row.get("id", "")),
                "reference": normalize_text(str(row.get("reference", ""))),
                "nbest": candidates,
                "nbest_pinyin": [joined_pinyin(item) for item in candidates],
                "candidate_scores": scores,
            }
            writer.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.flush()
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(json.dumps({"written": written, "rate": written / elapsed}, ensure_ascii=False), flush=True)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "written": written,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
