#!/usr/bin/env python3
"""Fetch MLS-English Parquet shards and retain only upstream CB-Whisper utterances."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


SHARD_COUNT = 1416
BASE_URL = "https://hf-mirror.com/datasets/parler-tts/mls_eng/resolve/main/data"
UPSTREAM_UTTID = Path("/root/autodl-tmp/datasets/mls/train/mls_english_opus/uttid")
OUTPUT = Path("/root/autodl-tmp/datasets/mls_english_opus_upstream_cbwhisper")
EXISTING_SHARD_ZERO = Path("/root/autodl-tmp/datasets/mls_hf_parquet/train-00000-of-01416.parquet")


def code_from_audio(audio: dict) -> str:
    return Path(audio["path"]).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=SHARD_COUNT)
    args = parser.parse_args()

    required = {
        line.strip().split()[0]
        for line in UPSTREAM_UTTID.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    audio_root = OUTPUT / "train" / "audio"
    temp_root = OUTPUT / ".shards"
    marker_root = OUTPUT / ".processed"
    for directory in (audio_root, temp_root, marker_root):
        directory.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    started = time.time()
    completed = 0
    found = {
        path.stem
        for path in audio_root.glob("*/*/*.opus")
        if path.stem in required
    }

    def process_shard(index: int) -> dict:
        marker = marker_root / f"{index:05d}.done"
        if marker.exists():
            return {"index": index, "status": "existing", "matches": 0, "bytes": 0}
        final_name = f"train-{index:05d}-of-{SHARD_COUNT:05d}.parquet"
        if index == 0 and EXISTING_SHARD_ZERO.exists():
            parquet_path = EXISTING_SHARD_ZERO
            remove_after = False
        else:
            parquet_path = temp_root / final_name
            remove_after = True
            url = f"{BASE_URL}/{final_name}"
            result = subprocess.run(
                [
                    "curl", "-L", "--fail", "--retry", "8", "--retry-all-errors",
                    "--retry-delay", "2", "-C", "-", "-o", str(parquet_path), url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"shard {index} download failed: {result.stderr[-500:]}")

        matches = []
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=128, columns=["audio", "speaker_id", "book_id"]):
            for row in batch.to_pylist():
                code = code_from_audio(row["audio"])
                if code not in required:
                    continue
                destination = audio_root / str(row["speaker_id"]) / str(row["book_id"]) / f"{code}.opus"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(row["audio"]["bytes"])
                matches.append(code)
        size = parquet_path.stat().st_size
        marker.write_text("\n".join(matches) + ("\n" if matches else ""), encoding="utf-8")
        if remove_after:
            parquet_path.unlink()
        return {"index": index, "status": "processed", "matches": len(matches), "codes": matches, "bytes": size}

    indices = list(range(max(0, args.start), min(SHARD_COUNT, args.end)))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_shard, index): index for index in indices}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                completed += 1
                found.update(result.get("codes", []))
                if completed % 10 == 0 or result.get("matches", 0) > 0:
                    elapsed = max(time.time() - started, 1.0)
                    rate = completed / elapsed
                    remaining = (len(indices) - completed) / max(rate, 1e-9)
                    progress = {
                        "completed_shards": completed,
                        "total_shards": len(indices),
                        "last_shard": result["index"],
                        "last_matches": result.get("matches", 0),
                        "found_utterances": len(found),
                        "required_utterances": len(required),
                        "shards_per_minute": rate * 60.0,
                        "eta_hours": remaining / 3600.0,
                    }
                    print(json.dumps(progress), flush=True)
                    (OUTPUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")

    missing = sorted(required - found)
    summary = {
        "required": len(required),
        "found": len(found),
        "missing": len(missing),
        "missing_preview": missing[:20],
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
