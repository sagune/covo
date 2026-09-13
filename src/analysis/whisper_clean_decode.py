import argparse
import csv
import os
import re
import sys
import unicodedata

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from data.dataset import AishellHotwordDataset
from data.data_collator import HotwordDataCollator


def _to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(str(text))
    except Exception:
        try:
            import zhconv

            return zhconv.convert(str(text), "zh-cn")
        except Exception:
            return str(text)


def _cn_digit_seq_to_str(s: str):
    table = {
        "零": "0", "〇": "0", "○": "0", "洞": "0",
        "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
        "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
    }
    out = []
    for ch in s:
        if ch not in table:
            return None
        out.append(table[ch])
    return "".join(out)


def _cn_with_units_to_int(s: str):
    digit = {
        "零": 0, "〇": 0, "○": 0, "洞": 0,
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    small = {"十": 10, "百": 100, "千": 1000}
    large = {"万": 10000, "亿": 100000000}
    total = section = number = 0
    seen_digit = False
    for ch in s:
        if ch in digit:
            number = digit[ch]
            seen_digit = True
        elif ch in small:
            if number == 0:
                number = 1
            section += number * small[ch]
            number = 0
        elif ch in large:
            section += number
            if section == 0:
                section = 1
            total += section * large[ch]
            section = number = 0
        else:
            return None
    if not seen_digit:
        return None
    return total + section + number


def _normalize_cn_numeric_chunk(s: str) -> str:
    if not s:
        return s
    if any(ch in "十百千万亿" for ch in s):
        val = _cn_with_units_to_int(s)
        return str(val) if val is not None else s
    val = _cn_digit_seq_to_str(s)
    return val if val is not None else s


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _to_simplified(text)
    text = re.sub(r"％", "%", text)
    units = "美元|元|块|港元|欧元|日元|人民币|%|％|厘米|米|公斤|千克|岁|届|次|项|家|名|位|人|所|个|小时|分钟|秒"
    text = re.sub(r"百分之([零〇○一二两三四五六七八九十百千万亿]+)", lambda m: _normalize_cn_numeric_chunk(m.group(1)) + "%", text)
    text = re.sub(r"百分之(\d+)", lambda m: str(int(m.group(1))) + "%", text)
    text = re.sub(r"([零〇○一二两三四五六七八九]{2,4})(?=年)", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)
    text = re.sub(r"([零〇○一二两三四五六七八九十]{1,3})(?=[月日号岁])", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)
    text = re.sub(rf"([零〇○一二两三四五六七八九十百千万亿]+)(?=({units}))", lambda m: _normalize_cn_numeric_chunk(m.group(1)), text)

    def repl_large(m):
        try:
            base = int(m.group(1))
        except Exception:
            return m.group(0)
        return str(base * {"万": 10000, "亿": 100000000}[m.group(2)])

    text = re.sub(r"(\d+)([万亿])(?=(美元|元|块|港元|欧元|日元|人民币))", repl_large, text)
    text = re.sub(r"\d+", lambda m: str(int(m.group(0))), text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"[·•‧・]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def edit_distance(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../datasets/aishell/data_aishell")
    parser.add_argument("--split", default="test")
    parser.add_argument("--kw-type", default="tts")
    parser.add_argument("--whisper-ckpt", default="openai/whisper-large-v3")
    parser.add_argument("--output-csv", default="logs/whisper_clean_decode.csv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    processor = WhisperProcessor.from_pretrained(args.whisper_ckpt, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_ckpt,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    dataset = AishellHotwordDataset(
        root=os.path.join(args.root, "hotword"),
        split=args.split,
        size=(150, 750),
        r1_only=False,
        hotwords_per_group=100,
        kw_type=args.kw_type,
        load_audio=True,
        wav_folder=os.path.join(args.root, "wav"),
        feature_extractor=processor.feature_extractor,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=HotwordDataCollator(), num_workers=4)

    rows = []
    total_err = 0
    total_ref = 0
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(loader, desc="standard whisper decode")):
            if args.limit > 0 and idx >= args.limit:
                break
            features = batch["utterance"]["features"].to(device=device, dtype=model.dtype)
            attention_mask = batch["utterance"].get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pred_ids = model.generate(
                input_features=features,
                attention_mask=attention_mask,
                language="zh",
                task="transcribe",
                return_timestamps=False,
                num_beams=args.num_beams,
                do_sample=False,
            )
            pred = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            ref = str(batch["transcript"])
            norm_pred = normalize_surface(pred)
            norm_ref = normalize_surface(ref)
            err = edit_distance(list(norm_ref), list(norm_pred))
            total_err += err
            total_ref += len(norm_ref)
            rows.append({
                "idx": idx,
                "ref": ref,
                "pred": pred,
                "norm_ref": norm_ref,
                "norm_pred": norm_pred,
                "edit_distance": err,
                "ref_len": len(norm_ref),
                "cer": err / max(len(norm_ref), 1),
            })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    corpus_cer = total_err / max(total_ref, 1)
    mean_cer = sum(r["cer"] for r in rows) / max(len(rows), 1)
    print({"rows": len(rows), "corpus_cer": corpus_cer, "mean_cer": mean_cer, "output_csv": args.output_csv})


if __name__ == "__main__":
    main()
