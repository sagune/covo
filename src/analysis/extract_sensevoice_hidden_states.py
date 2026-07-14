#!/usr/bin/env python
"""Extract SenseVoice encoder hidden states for CB-style KWS training.

The output format intentionally matches ``src/hs_utils.py`` so existing KWS
datasets can load the generated ``.bin`` files without a new data loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hs_utils import quantize_hidden_states  # noqa: E402


def read_codes(path: str | None) -> set[str] | None:
    if path is None or str(path).strip() == "":
        return None
    codes = set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            codes.add(line.split("\t")[0].strip().split(" ")[0].strip())
    return codes


def iter_audio_files(root: str) -> Dict[str, str]:
    patterns = ["*.wav", "*.mp3", "*.opus", "*/*.wav", "*/*.mp3", "*/*.opus", "*/*/*.wav", "*/*/*.mp3", "*/*/*.opus"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(os.path.join(root, pattern)))
    output = {}
    for file_name in sorted(set(files)):
        code = os.path.splitext(os.path.basename(file_name))[0]
        if code.startswith("audio-"):
            code = code[6:]
        output[code] = file_name
    return output


def build_model(model_name: str, device: str):
    from funasr import AutoModel

    return AutoModel(model=model_name, device=device, disable_update=True)


def load_fbank(auto_model: Any, audio_file: str, device: str, audio_fs: int = 16000):
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    frontend = auto_model.kwargs["frontend"]
    audio = load_audio_text_image_video(
        audio_file,
        fs=frontend.fs,
        audio_fs=audio_fs,
        data_type="sound",
        tokenizer=auto_model.kwargs.get("tokenizer"),
    )
    speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
    return speech.to(device), speech_lengths.to(device)


def encode_sensevoice(
    auto_model: Any,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    language: str,
    textnorm: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = auto_model.model
    device = speech.device
    language_query = model.embed(
        torch.LongTensor([[model.lid_dict[language] if language in model.lid_dict else 0]]).to(device)
    ).repeat(speech.size(0), 1, 1)
    textnorm_query = model.embed(torch.LongTensor([[model.textnorm_dict[textnorm]]]).to(device)).repeat(
        speech.size(0), 1, 1
    )
    speech = torch.cat((textnorm_query, speech), dim=1)
    speech_lengths = speech_lengths + 1
    event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(device)).repeat(speech.size(0), 1, 1)
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths = speech_lengths + 3
    encoder_out, encoder_out_lens = model.encoder(speech, speech_lengths)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    return encoder_out, encoder_out_lens


def normalize_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states / torch.linalg.norm(hidden_states, dim=-1, keepdim=True).clamp_min(1e-8)


def write_manifest(
    target: str,
    manifest_name: str,
    model_name: str,
    auto_model: Any,
    language: str,
    textnorm: str,
    device: str,
    strip_query_tokens: bool,
) -> None:
    if not manifest_name:
        return
    model = auto_model.model
    manifest = {
        "format_version": 1,
        "model": model_name,
        "model_class": type(model).__name__,
        "frontend": type(auto_model.kwargs.get("frontend")).__name__,
        "input_size": int(auto_model.kwargs.get("input_size", 0) or 0),
        "encoder_output_size": int(getattr(model, "encoder_output_size", 0) or 0),
        "language": language,
        "textnorm": textnorm,
        "normalized": True,
        "quantized": "int8_symmetric_127",
        "strip_query_tokens": bool(strip_query_tokens),
        "query_tokens": 4,
        "device": device,
    }
    path = Path(target) / manifest_name
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract(args: argparse.Namespace) -> int:
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    codes = read_codes(args.utterances)
    audio_files = iter_audio_files(args.audios)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    auto_model = build_model(args.model, device=device)
    auto_model.model.eval()
    write_manifest(
        target=str(target),
        manifest_name=args.manifest_name,
        model_name=args.model,
        auto_model=auto_model,
        language=args.language,
        textnorm=args.textnorm,
        device=device,
        strip_query_tokens=not args.keep_query_tokens,
    )
    print(
        json.dumps(
            {
                "event": "sensevoice_extract_start",
                "audios": args.audios,
                "target": str(target),
                "files": len(audio_files),
                "codes_filter": len(codes) if codes is not None else None,
                "device": device,
                "model": args.model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    written = 0
    skipped = 0
    failed: List[Dict[str, str]] = []
    for code, audio_file in tqdm(audio_files.items()):
        if codes is not None and code not in codes:
            continue
        out_file = target / f"{code}.bin"
        if args.skip_existing and out_file.exists():
            skipped += 1
            continue
        try:
            speech, speech_lengths = load_fbank(auto_model, audio_file, device=device, audio_fs=int(args.audio_fs))
            with torch.inference_mode():
                encoder_out, encoder_out_lens = encode_sensevoice(
                    auto_model,
                    speech=speech,
                    speech_lengths=speech_lengths,
                    language=args.language,
                    textnorm=args.textnorm,
                )
            valid_len = int(encoder_out_lens[0].item())
            hs = encoder_out[:, :valid_len, :]
            if not args.keep_query_tokens:
                hs = hs[:, 4:, :]
            hs = normalize_hidden_states(hs)
            with out_file.open("wb") as handle:
                torch.save(quantize_hidden_states(hs.detach().cpu()), handle)
            written += 1
        except Exception as exc:
            failed.append({"code": code, "audio": audio_file, "error": str(exc)})
            print(f"[sensevoice_extract][warn] {code} {audio_file}: {exc}", flush=True)
            if args.fail_on_error:
                raise

    summary = {"written": written, "skipped": skipped, "failed": len(failed)}
    print(json.dumps({"event": "sensevoice_extract_done", **summary}, ensure_ascii=False), flush=True)
    if failed:
        print(json.dumps({"first_failures": failed[:5]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failed or not args.fail_on_error else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--audios", required=True)
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("-u", "--utterances", default="")
    parser.add_argument("-m", "--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--textnorm", choices=["withitn", "woitn"], default="withitn")
    parser.add_argument("--audio-fs", type=int, default=16000)
    parser.add_argument("--manifest-name", default="_hs_manifest.json")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-query-tokens", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    return extract(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
