import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


TIME_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)
TEXT_CLEAN_RE = re.compile(r"[\s\u200b]+")


@dataclass
class Segment:
    code: str
    source: str
    start: float
    end: float
    text: str


def parse_time(match: re.Match, prefix: str) -> float:
    hours = int(match.group(prefix + "h"))
    minutes = int(match.group(prefix + "m"))
    seconds = int(match.group(prefix + "s"))
    millis = int(match.group(prefix + "ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = TEXT_CLEAN_RE.sub("", text)
    return text.strip()


def read_srt(path: Path, bac_prefix: str, speaker: str) -> list[Segment]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    segments = []
    kept_idx = 0
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_line_idx = next((idx for idx, line in enumerate(lines) if TIME_RE.search(line)), None)
        if time_line_idx is None:
            continue
        match = TIME_RE.search(lines[time_line_idx])
        start = parse_time(match, "s")
        end = parse_time(match, "e")
        text = clean_text("".join(lines[time_line_idx + 1 :]))
        if not text or end <= start:
            continue
        kept_idx += 1
        code = f"{bac_prefix}{speaker}W{kept_idx:04d}"
        segments.append(Segment(code=code, source=path.stem, start=start, end=end, text=text))
    return segments


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_audio(video: Path, out_wav: Path) -> None:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_wav),
    ])


def cut_segment(full_wav: Path, out_wav: Path, start: float, end: float) -> None:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(full_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_wav),
    ])


def copy_hotword_assets(source_root: Path, target_root: Path) -> list[str]:
    src_split = source_root / "hotword" / "test"
    dst_split = target_root / "hotword" / "test"
    dst_split.mkdir(parents=True, exist_ok=True)

    for name in ["hotword.txt", "hotword_voice.txt", "r1-hotword.txt"]:
        src = src_split / name
        if src.exists():
            shutil.copy2(src, dst_split / name)

    for name in ["keywords-audios", "keywords-hs"]:
        src = src_split / name
        dst = dst_split / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    with open(dst_split / "hotword.txt", "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def write_metadata(target_root: Path, segments: list[Segment], hotwords: list[str]) -> None:
    split = target_root / "hotword" / "test"
    split.mkdir(parents=True, exist_ok=True)
    (split / "text").write_text(
        "\n".join(f"{seg.code} {seg.text}" for seg in segments) + "\n",
        encoding="utf-8",
    )
    (split / "uttid").write_text(
        "\n".join(f"{seg.code} {seg.text}" for seg in segments) + "\n",
        encoding="utf-8",
    )

    aligned_rows = []
    for seg in segments:
        duration = max(seg.end - seg.start, 1e-3)
        text_len = max(len(seg.text), 1)
        for hotword in hotwords:
            pos = seg.text.find(hotword)
            if pos < 0:
                continue
            start = duration * pos / text_len
            end = duration * (pos + len(hotword)) / text_len
            aligned_rows.append(f"{hotword}\t{seg.code}\t{start:.3f}\t{end:.3f}")
    (split / "aligned.txt").write_text("\n".join(aligned_rows) + ("\n" if aligned_rows else ""), encoding="utf-8")

    summary = {
        "segments": len(segments),
        "aligned_hotword_mentions": len(aligned_rows),
        "sources": sorted({seg.source for seg in segments}),
    }
    (target_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/newdata", help="folder containing extracted video folders")
    parser.add_argument("--target", default="/root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3")
    parser.add_argument("--hotword-source", default="/root/autodl-tmp/datasets/shuili/data_shuil_largev3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    if args.force and target.exists():
        shutil.rmtree(target)

    hotwords = copy_hotword_assets(Path(args.hotword_source), target)
    full_audio_dir = target / "source_audio"
    all_segments: list[Segment] = []

    video_specs = [
        ("7月11日", "BAC011", "S0001"),
        ("7月12日", "BAC012", "S0002"),
    ]
    for folder, bac_prefix, speaker in video_specs:
        video = source / folder / f"{folder}.mp4"
        srt = source / folder / f"{folder}.srt"
        if not video.exists() or not srt.exists():
            raise FileNotFoundError(f"missing video or srt for {folder}")
        full_wav = full_audio_dir / f"{folder}.wav"
        extract_audio(video, full_wav)
        segments = read_srt(srt, bac_prefix=bac_prefix, speaker=speaker)
        for seg in segments:
            out_wav = target / "wav" / "test" / speaker / f"{seg.code}.wav"
            cut_segment(full_wav, out_wav, seg.start, seg.end)
        all_segments.extend(segments)

    write_metadata(target, all_segments, hotwords)
    print(json.dumps({
        "target": str(target),
        "segments": len(all_segments),
        "hotwords": len(hotwords),
        "summary": str(target / "build_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
