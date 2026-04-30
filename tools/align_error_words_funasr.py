import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torchaudio
from funasr import AutoModel


PUNCT_RE = re.compile(r"[\s\.,!?;:'\"，。！？；：、“”‘’（）()\[\]【】<>《》—…·]+")


def read_text(path: str) -> str:
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub("", text)


def extract_char_timestamps(asr_item: Dict) -> Tuple[str, List[Tuple[float, float]]]:
    text = asr_item.get("text", "")
    ts = (
        asr_item.get("timestamp")
        or asr_item.get("timestamps")
        or asr_item.get("word_timestamp")
        or asr_item.get("char_timestamp")
    )

    if ts is None:
        raise ValueError("ASR result does not contain timestamps")

    # Normalize timestamp list to list of [start, end]
    if isinstance(ts, dict) and "time_stamp" in ts:
        ts = ts["time_stamp"]

    char_times: List[Tuple[float, float]] = []
    if isinstance(ts, list) and len(ts) > 0 and isinstance(ts[0], dict):
        for item in ts:
            start = float(item.get("start", item.get("begin", 0.0)))
            end = float(item.get("end", item.get("finish", 0.0)))
            char_times.append((start, end))
    elif isinstance(ts, list):
        for pair in ts:
            if pair is None or len(pair) < 2:
                char_times.append((0.0, 0.0))
            else:
                char_times.append((float(pair[0]), float(pair[1])))
    else:
        raise ValueError("Unrecognized timestamp format")

    # Filter punctuation in text and timestamps together
    filtered_text = []
    filtered_times = []
    for ch, t in zip(text, char_times):
        if PUNCT_RE.match(ch):
            continue
        filtered_text.append(ch)
        filtered_times.append(t)

    return "".join(filtered_text), filtered_times


def align_ref_to_asr(ref: str, hyp: str) -> List[Optional[int]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            options = [
                (dp[i - 1][j - 1] + cost, ("sub" if cost else "match", i - 1, j - 1)),
                (dp[i - 1][j] + 1, ("del", i - 1, j)),
                (dp[i][j - 1] + 1, ("ins", i, j - 1)),
            ]
            dp[i][j], bt[i][j] = min(options, key=lambda x: x[0])

    mapping: List[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        action, pi, pj = bt[i][j]
        if action in ("match", "sub"):
            mapping[pi] = pj
            i, j = pi, pj
        elif action == "del":
            i, j = pi, pj
        else:
            i, j = pi, pj

    return mapping


def load_per_file_errors(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors_by_clip: Dict[str, List[str]] = {}
    for item in data:
        audio_path = item.get("audio_path", "")
        txt_path = item.get("txt_path", "")
        clip_name = os.path.basename(audio_path) if audio_path else os.path.basename(txt_path).replace("transcript_", "clip_").replace(".txt", ".wav")
        errors_by_clip[clip_name] = item.get("reference_error_words", []) or []

    return errors_by_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with clip_*.wav and transcript_*.txt")
    parser.add_argument("--output_dir", required=True, help="Folder with per_file_results.json")
    parser.add_argument("--per_file_results", default=None, help="Path to per_file_results.json")
    parser.add_argument("--output_file", default="error_word_timestamps.txt")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model", default="paraformer-zh", help="FunASR model name")
    parser.add_argument("--vad_model", default="fsmn-vad", help="FunASR VAD model")
    parser.add_argument("--punc_model", default="ct-punc", help="FunASR punctuation model")
    parser.add_argument("--model_revision", default="v2.0.4", help="FunASR model revision")
    args = parser.parse_args()

    per_file_results = args.per_file_results or os.path.join(args.output_dir, "per_file_results.json")
    errors_by_clip = load_per_file_errors(per_file_results)

    model = AutoModel(
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        model_revision=args.model_revision,
        device=args.device
    )

    output_path = os.path.join(args.output_dir, args.output_file)
    missing_path = os.path.join(args.output_dir, "error_word_timestamps.missing.txt")

    with open(output_path, "w", encoding="utf-8") as out_f, open(missing_path, "w", encoding="utf-8") as miss_f:
        for clip_name, error_words in errors_by_clip.items():
            if not error_words:
                continue

            clip_path = os.path.join(args.data_dir, clip_name)
            transcript_path = os.path.join(args.data_dir, clip_name.replace("clip_", "transcript_").replace(".wav", ".txt"))
            if not os.path.exists(clip_path) or not os.path.exists(transcript_path):
                miss_f.write(f"{clip_name}\tMISSING_FILES\n")
                continue

            reference_text = read_text(transcript_path)
            ref_norm = normalize_text(reference_text)

            asr_res = model.generate(input=clip_path, return_timestamp=True)
            asr_item = asr_res[0]
            hyp_norm, hyp_times = extract_char_timestamps(asr_item)

            mapping = align_ref_to_asr(ref_norm, hyp_norm)
            ref_times: List[Optional[Tuple[float, float]]] = [None] * len(ref_norm)
            for i, j in enumerate(mapping):
                if j is not None and j < len(hyp_times):
                    ref_times[i] = hyp_times[j]

            word_positions: Dict[str, List[int]] = defaultdict(list)
            for word in error_words:
                word_norm = normalize_text(word)
                if not word_norm:
                    continue
                if word_norm not in word_positions:
                    start = 0
                    while True:
                        idx = ref_norm.find(word_norm, start)
                        if idx == -1:
                            break
                        word_positions[word_norm].append(idx)
                        start = idx + 1

            word_cursor = defaultdict(int)
            for word in error_words:
                word_norm = normalize_text(word)
                positions = word_positions.get(word_norm, [])
                cursor = word_cursor[word_norm]
                if cursor >= len(positions):
                    miss_f.write(f"{word}\t{clip_name}\tNOT_FOUND_IN_TRANSCRIPT\n")
                    continue
                start_idx = positions[cursor]
                end_idx = start_idx + len(word_norm) - 1
                word_cursor[word_norm] += 1

                span_times = [t for t in ref_times[start_idx:end_idx + 1] if t is not None]
                if not span_times:
                    miss_f.write(f"{word}\t{clip_name}\tNO_TIMESTAMP\n")
                    continue
                start_time = min(t[0] for t in span_times)
                end_time = max(t[1] for t in span_times)
                out_f.write(f"{word}\t{clip_name}\t{start_time:.3f}\t{end_time:.3f}\n")


if __name__ == "__main__":
    main()
