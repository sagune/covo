#!/usr/bin/env python3
"""Synthesize indexed keyword audio with bounded Edge-TTS concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import edge_tts


DEFAULT_VOICES = (
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-YunxiaNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunyangNeural",
)


async def synthesize_one(
    semaphore: asyncio.Semaphore,
    text: str,
    voice: str,
    output: Path,
    retries: int,
) -> tuple[bool, str]:
    if output.exists() and output.stat().st_size > 0:
        return True, "existing"
    async with semaphore:
        last_error = ""
        for attempt in range(1, retries + 1):
            temporary = output.with_suffix(f".attempt{attempt}.mp3")
            try:
                await edge_tts.Communicate(text, voice).save(str(temporary))
                if temporary.exists() and temporary.stat().st_size > 0:
                    temporary.replace(output)
                    return True, "written"
                last_error = "empty output"
            except Exception as exc:
                last_error = str(exc)
            if temporary.exists():
                temporary.unlink()
            await asyncio.sleep(min(attempt * 2, 10))
        return False, last_error


async def run(args: argparse.Namespace) -> int:
    lines = [line for line in args.keywords.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.limit is not None:
        lines = lines[:max(0, args.limit)]
    width = len(str(max(len(lines) - 1, 0)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    jobs = []
    for index, line in enumerate(lines):
        parts = line.split("\t")
        text = parts[0].strip()
        voice = parts[1].strip() if len(parts) > 1 and parts[1].strip() else DEFAULT_VOICES[index % len(DEFAULT_VOICES)]
        jobs.append(synthesize_one(
            semaphore,
            text=text,
            voice=voice,
            output=args.output_dir / f"{index:0{width}d}.mp3",
            retries=max(1, args.retries),
        ))

    written = existing = failed = 0
    for completed, result in enumerate(asyncio.as_completed(jobs), 1):
        ok, status = await result
        if ok and status == "written":
            written += 1
        elif ok:
            existing += 1
        else:
            failed += 1
            print(f"[edge-tts][warn] {status}", flush=True)
        if args.log_every > 0 and completed % args.log_every == 0:
            print(json.dumps({"completed": completed, "written": written, "existing": existing, "failed": failed}), flush=True)

    print(json.dumps({
        "keywords": len(lines),
        "written": written,
        "existing": existing,
        "failed": failed,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False), flush=True)
    return 1 if failed and args.fail_on_error else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
