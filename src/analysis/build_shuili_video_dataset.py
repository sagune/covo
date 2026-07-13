import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


TIME_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)
TEXT_CLEAN_RE = re.compile(r"[\s\u200b]+")

DOMAIN_CHARS = set(
    "水河江湖海库坝闸渠灌排洪涝旱堤泵站溉流域资源工程调度输丹湖北汉长黄生态治理防供引清淤蓄泄"
)
DOMAIN_WORDS = (
    "水资源", "水库", "水利", "工程", "河流", "河道", "江河", "湖泊", "流域", "灌区", "渠道",
    "灌溉", "防洪", "排涝", "供水", "输水", "调水", "南水北调", "丹江口", "汉江", "长江",
    "生态", "水质", "水量", "水位", "水电", "泵站", "水闸", "堤防", "大坝", "水厂", "涵闸",
)
STOP_TERMS = {
    "这个", "这一个", "那个", "那么", "我们", "咱们", "他们", "它们", "一个", "一种", "一些", "不是",
    "就是", "可以", "进行", "通过", "因为", "所以", "然后", "如果", "时候", "现在", "已经", "没有",
    "这里", "大家", "来说", "起来", "出来", "不断", "作为", "成为", "对于", "以及", "还有", "开始",
    "看到", "今天", "同时", "为了", "更加", "非常", "重要", "问题", "方式", "过程", "形成", "发展",
}
BAD_HOTWORD_SUBSTRINGS = (
    "的", "了", "着", "和", "与", "把", "就", "这个", "我们", "咱们", "他们", "它们",
    "给", "等", "有", "是", "在", "成为", "用好", "最多", "当地", "安全", "有限",
    "薄", "方米", "系列",
)


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


def is_good_hotword(term: str) -> bool:
    if term in STOP_TERMS:
        return False
    if len(term) < 2 or len(term) > 8:
        return False
    if any(ch.isdigit() or ("a" <= ch.lower() <= "z") for ch in term):
        return False
    if term.startswith(("的", "了", "和", "与", "在", "对", "从", "把", "将", "为")):
        return False
    if term.endswith(("的", "了", "呢", "啊", "嘛", "吗", "着", "过", "和", "与", "在", "是")):
        return False
    if term not in DOMAIN_WORDS and any(item in term for item in BAD_HOTWORD_SUBSTRINGS):
        return False
    return any(ch in DOMAIN_CHARS for ch in term) or any(word in term for word in DOMAIN_WORDS)


def extract_hotword_candidates(texts: list[str], old_hotwords: list[str], max_hotwords: int) -> list[str]:
    old_set = set(old_hotwords)
    counter: Counter[str] = Counter()
    for text in texts:
        chars = clean_text(text)
        for size in range(2, 7):
            for idx in range(0, max(0, len(chars) - size + 1)):
                term = chars[idx : idx + size]
                if is_good_hotword(term):
                    counter[term] += 1
    for text in texts:
        for word in DOMAIN_WORDS:
            if word in text and is_good_hotword(word):
                counter[word] += 3

    def score(item: tuple[str, int]) -> tuple[float, int, str]:
        term, freq = item
        bonus = 0
        if any(word in term for word in DOMAIN_WORDS):
            bonus += 8
        if term.endswith(("工程", "水库", "水利", "河道", "渠道", "流域", "水资源", "调水", "供水", "防洪")):
            bonus += 5
        return (freq * (len(term) + 0.8) + bonus, len(term), term)

    ranked = [term for term, _ in sorted(counter.items(), key=score, reverse=True)]
    output = list(old_hotwords)
    for term in ranked:
        if len(output) >= max_hotwords:
            break
        if term in old_set:
            continue
        if any(term != kept and term in kept and len(term) <= 2 for kept in output):
            continue
        output.append(term)
        old_set.add(term)
    return output


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


def copy_hotword_assets(source_root: Path, target_root: Path, hotwords: list[str] | None = None) -> list[str]:
    src_split = source_root / "hotword" / "test"
    dst_split = target_root / "hotword" / "test"
    dst_split.mkdir(parents=True, exist_ok=True)

    for name in ["hotword_voice.txt", "r1-hotword.txt"]:
        src = src_split / name
        if src.exists():
            shutil.copy2(src, dst_split / name)

    old_hotwords = [
        line.strip()
        for line in (src_split / "hotword.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    selected = hotwords if hotwords is not None else old_hotwords
    (dst_split / "hotword.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

    for name in ["keywords-audios", "keywords-hs"]:
        dst = dst_split / name
        if dst.exists():
            shutil.rmtree(dst)
        for subtype in ["natural", "tts"]:
            (dst / subtype).mkdir(parents=True, exist_ok=True)

    old_index = {word: idx for idx, word in enumerate(old_hotwords)}
    old_zfill = len(str(len(old_hotwords) - 1))
    new_zfill = len(str(len(selected) - 1))
    for new_idx, word in enumerate(selected):
        if word not in old_index:
            continue
        old_idx = old_index[word]
        for name in ["keywords-audios", "keywords-hs"]:
            for subtype in ["natural", "tts"]:
                src = src_split / name / subtype / f"{old_idx:0{old_zfill}d}.bin"
                suffix = ".bin"
                if name == "keywords-audios":
                    wav_src = src_split / name / subtype / f"{old_idx:0{old_zfill}d}.wav"
                    mp3_src = src_split / name / subtype / f"{old_idx:0{old_zfill}d}.mp3"
                    src = wav_src if wav_src.exists() else mp3_src
                    suffix = src.suffix
                if src.exists():
                    shutil.copy2(src, dst_split / name / subtype / f"{new_idx:0{new_zfill}d}{suffix}")
    return selected


def write_keyword_audio_for_new_terms(
    target_root: Path,
    segments: list[Segment],
    hotwords: list[str],
    old_hotwords: set[str],
) -> int:
    split = target_root / "hotword" / "test"
    out_dir = split / "keywords-audios" / "natural"
    out_dir.mkdir(parents=True, exist_ok=True)
    zfill = len(str(len(hotwords) - 1))
    by_code = {seg.code: seg for seg in segments}
    written = 0
    for idx, hotword in enumerate(hotwords):
        if hotword in old_hotwords:
            continue
        existing = out_dir / f"{idx:0{zfill}d}.wav"
        if existing.exists() and existing.stat().st_size > 0:
            continue
        match_seg = next((seg for seg in segments if hotword in seg.text), None)
        if match_seg is None:
            continue
        duration = max(match_seg.end - match_seg.start, 0.3)
        pos = max(match_seg.text.find(hotword), 0)
        text_len = max(len(match_seg.text), 1)
        local_start = max(0.0, duration * pos / text_len - 0.05)
        local_end = min(duration, duration * (pos + len(hotword)) / text_len + 0.05)
        if local_end - local_start < 0.18:
            center = (local_start + local_end) / 2.0
            local_start = max(0.0, center - 0.09)
            local_end = min(duration, center + 0.09)
        src_wav = target_root / "wav" / "test" / match_seg.code[6:11] / f"{match_seg.code}.wav"
        if not src_wav.exists():
            continue
        cut_segment(src_wav, existing, local_start, local_end)
        written += 1
    return written


def write_metadata(target_root: Path, segments: list[Segment], hotwords: list[str], keyword_audio_written: int = 0) -> None:
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
        "hotwords": len(hotwords),
        "keyword_audio_written": keyword_audio_written,
        "sources": sorted({seg.source for seg in segments}),
    }
    (target_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/newdata", help="folder containing extracted video folders")
    parser.add_argument("--target", default="/root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3")
    parser.add_argument("--hotword-source", default="/root/autodl-tmp/datasets/shuili/data_shuil_largev3")
    parser.add_argument("--expand-hotwords", action="store_true")
    parser.add_argument("--max-hotwords", type=int, default=360)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    if args.force and target.exists():
        shutil.rmtree(target)

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

    src_hotwords = [
        line.strip()
        for line in (Path(args.hotword_source) / "hotword" / "test" / "hotword.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    hotwords = extract_hotword_candidates(
        texts=[seg.text for seg in all_segments],
        old_hotwords=src_hotwords,
        max_hotwords=max(len(src_hotwords), int(args.max_hotwords)),
    ) if args.expand_hotwords else src_hotwords
    hotwords = copy_hotword_assets(Path(args.hotword_source), target, hotwords=hotwords)
    keyword_audio_written = write_keyword_audio_for_new_terms(
        target_root=target,
        segments=all_segments,
        hotwords=hotwords,
        old_hotwords=set(src_hotwords),
    ) if args.expand_hotwords else 0
    write_metadata(target, all_segments, hotwords, keyword_audio_written=keyword_audio_written)
    print(json.dumps({
        "target": str(target),
        "segments": len(all_segments),
        "hotwords": len(hotwords),
        "keyword_audio_written": keyword_audio_written,
        "summary": str(target / "build_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
