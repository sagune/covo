#!/usr/bin/env python3
"""Attach forced-CTC acoustic scores to existing SenseVoice N-best records."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.generate_sensevoice_chinesehp import encode_audio, normalize_text


def ctc_sequence_score(log_probs: torch.Tensor, token_ids: List[int], blank_id: int) -> float:
    """Return the full CTC log likelihood of one fixed token sequence."""
    if log_probs.dim() != 2:
        raise ValueError(f"expected [time,vocab] log probabilities, got {tuple(log_probs.shape)}")
    if not token_ids:
        return -1e9
    device = log_probs.device
    targets = torch.tensor(token_ids, dtype=torch.long, device=device)
    input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long, device=device)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    loss = F.ctc_loss(
        log_probs.unsqueeze(1),
        targets,
        input_lengths,
        target_lengths,
        blank=int(blank_id),
        reduction="none",
        zero_infinity=False,
    )[0]
    score = -float(loss.detach().cpu())
    return score if math.isfinite(score) else -1e9


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in _read_jsonl(path):
        completed.add(str(row.get("id", "")))
    return completed


def _audio_map(path: Path) -> Dict[str, str]:
    output = {}
    for row in _read_jsonl(path):
        record_id = str(row.get("id", ""))
        wav = str(row.get("wav", ""))
        if record_id and wav:
            output[record_id] = wav
    return output


def _candidate_tokens(tokenizer: Any, text: str, excluded: set[int], blank_id: int) -> List[int]:
    return [
        int(token)
        for token in tokenizer.encode(str(text))
        if int(token) not in excluded and int(token) != int(blank_id)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing ChineseHP/Qwen N-best JSONL")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL with id, reference and wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--use-itn", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    from funasr import AutoModel

    wrapper = AutoModel(model=args.model, device=args.device, disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = next(model.parameters()).device
    vocab_size = int(model.ctc.ctc_lo.out_features)
    blank_id = int(model.blank_id)
    excluded = set(range(1, 16)) | set(range(25000, vocab_size))

    audio_by_id = _audio_map(args.manifest)
    completed = _completed_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for row_index, row in enumerate(_read_jsonl(args.input)):
        record_id = str(row.get("id", ""))
        if row_index % args.num_shards != args.shard_index:
            continue
        if not record_id or record_id in completed or record_id not in audio_by_id:
            continue
        selected.append(row)
        if args.limit and len(selected) >= int(args.limit):
            break

    started = time.monotonic()
    written = failed = 0
    with args.output.open("a", encoding="utf-8") as writer:
        for row in selected:
            record_id = str(row.get("id", ""))
            input_block = row.get("input", {}) or {}
            nbest = [
                str(item).strip()
                for item in input_block.get("nbest", row.get("nbest", [])) or []
                if str(item).strip()
            ][: max(1, int(args.max_nbest))]
            if not nbest:
                failed += 1
                continue
            wav = Path(audio_by_id[record_id])
            if not wav.is_absolute():
                wav = REPO_ROOT / wav
            try:
                log_probs = encode_audio(model, tokenizer, frontend, str(wav), device, args.use_itn)
                candidate_rows = []
                for rank, text in enumerate(nbest, 1):
                    token_ids = _candidate_tokens(tokenizer, text, excluded, blank_id)
                    acoustic_score = ctc_sequence_score(log_probs, token_ids, blank_id)
                    candidate_rows.append(
                        {
                            "rank": rank,
                            "text": text,
                            "token_ids": token_ids,
                            "asr_score": acoustic_score,
                            "search_score": acoustic_score,
                            "ctc_hotword_score": 0.0,
                            "source": "sensevoice_forced_ctc",
                            "baseline_candidate": rank == 1,
                        }
                    )
                output = {
                    "id": record_id,
                    "source": "sensevoice_forced_ctc",
                    "dataset": "stcmds",
                    "split": str(row.get("split", "train")),
                    "reference": normalize_text(str(row.get("reference", ""))),
                    "input": {
                        "asr_top1": nbest[0],
                        "nbest": nbest,
                        "nbest_pinyin": list(input_block.get("nbest_pinyin", []) or [])[: len(nbest)],
                        "hotwords": list(input_block.get("hotwords", []) or []),
                        "cbwhisper": {
                            "candidate_count": len(candidate_rows),
                            "candidates": candidate_rows,
                            "asr_backend": "sensevoice",
                            "score_type": "forced_ctc_log_likelihood",
                        },
                    },
                }
                writer.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                if written % 20 == 0:
                    writer.flush()
            except Exception as exc:
                failed += 1
                print(
                    json.dumps({"id": record_id, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                    flush=True,
                )
            if args.progress_every and (written + failed) % int(args.progress_every) == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "processed": written + failed,
                            "written": written,
                            "failed": failed,
                            "rate": (written + failed) / elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        writer.flush()
    print(
        json.dumps(
            {
                "input": str(args.input),
                "manifest": str(args.manifest),
                "output": str(args.output),
                "selected": len(selected),
                "written": written,
                "failed": failed,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
