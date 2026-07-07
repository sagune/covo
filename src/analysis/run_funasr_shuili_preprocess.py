#!/usr/bin/env python
"""Run FunASR models on Shuili wavs and export preprocessing transcripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


TAG_RE = re.compile(r"<\|[^|]+\|>")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def strip_asr_markup(text: Any) -> str:
    text = TAG_RE.sub("", str(text or ""))
    text = re.sub(r"\s+", "", text)
    return text.strip()


def normalize_for_cer(text: Any) -> str:
    text = strip_asr_markup(text)
    try:
        from opencc import OpenCC

        text = OpenCC("t2s").convert(text)
    except Exception:
        pass
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isdigit() or ("a" <= ch.lower() <= "z"))


def edit_distance(left: str, right: str) -> int:
    left = normalize_for_cer(left)
    right = normalize_for_cer(right)
    previous = list(range(len(right) + 1))
    for i, char_l in enumerate(left, 1):
        current = [i] + [0] * len(right)
        for j, char_r in enumerate(right, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if char_l == char_r else 1),
            )
        previous = current
    return previous[-1]


def extract_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return strip_asr_markup(first.get("text", ""))
    if isinstance(result, dict):
        return strip_asr_markup(result.get("text", ""))
    return strip_asr_markup(result)


def build_model(args: argparse.Namespace):
    from funasr import AutoModel

    kwargs = {"model": args.model, "device": args.device, "disable_update": True}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.vad_model:
        kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        kwargs["punc_model"] = args.punc_model
    return AutoModel(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_20260704.jsonl")
    parser.add_argument("--wav-root", default="datasets/shuili/data_shuil_largev3/wav/test/S0001")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--language", default="")
    parser.add_argument("--use-itn", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vad-model", default="")
    parser.add_argument("--punc-model", default="")
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.input)))
    if args.limit:
        rows = rows[: int(args.limit)]

    model = build_model(args)
    wav_root = Path(args.wav_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_edits = 0
    total_chars = 0
    exact = 0
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            utt_id = str(row.get("id", ""))
            wav = wav_root / f"{utt_id}.wav"
            generate_kwargs: Dict[str, Any] = {
                "input": str(wav),
                "batch_size_s": int(args.batch_size_s),
            }
            if args.language:
                generate_kwargs["language"] = args.language
            if args.use_itn:
                generate_kwargs["use_itn"] = True
            result = model.generate(**generate_kwargs)
            hyp = extract_text(result)
            ref = str(row.get("reference", ""))
            ref_norm = normalize_for_cer(ref)
            edits = edit_distance(ref, hyp)
            total_edits += edits
            total_chars += len(ref_norm)
            exact += int(edits == 0)
            out = {
                "id": utt_id,
                "reference": ref,
                "wav": str(wav),
                "model": args.model_label or args.model,
                "prediction": hyp,
                "normalized_prediction": normalize_for_cer(hyp),
                "edits": edits,
                "ref_chars": len(ref_norm),
            }
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "model": args.model,
        "model_label": args.model_label or args.model,
        "rows": written,
        "cer": total_edits / max(total_chars, 1),
        "edits": total_edits,
        "ref_chars": total_chars,
        "exact": exact,
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
