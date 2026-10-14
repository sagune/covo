import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from data.dataset import AishellHotwordDataset
from model.model import KWSModel


def _pinyin_units(text: str) -> List[str]:
    try:
        from pypinyin import lazy_pinyin

        return [str(x) for x in lazy_pinyin(str(text))]
    except Exception:
        return list(str(text))


def _sort_keywords_by_score(keywords: Iterable[str], scores: Dict[str, float]) -> List[str]:
    uniq = list(dict.fromkeys(str(k) for k in keywords if str(k).strip()))
    uniq.sort(key=lambda kw: float(scores.get(kw, 0.0)), reverse=True)
    return uniq


def _promote_nested_keywords(
    ranked_keywords: Sequence[str],
    selected_keywords: Sequence[str],
    scores: Dict[str, float],
    max_k: int,
    score_ratio: float,
    min_long_chars: int,
    max_extra: int,
) -> List[str]:
    if len(ranked_keywords) == 0 or len(selected_keywords) == 0 or max_k <= 0:
        return list(selected_keywords)[:max_k]

    selected = list(dict.fromkeys(selected_keywords))[:max_k]
    selected_set = set(selected)
    units = {kw: _pinyin_units(kw) for kw in ranked_keywords}
    promotions: List[Tuple[float, str]] = []

    for short_kw in selected:
        short_units = units.get(short_kw, _pinyin_units(short_kw))
        if len(short_kw) == 0 or len(short_units) == 0:
            continue
        short_score = float(scores.get(short_kw, 0.0))
        for long_kw in ranked_keywords:
            if long_kw in selected_set or long_kw == short_kw:
                continue
            long_units = units.get(long_kw, [])
            if len(long_kw) < min_long_chars or len(long_units) <= len(short_units):
                continue
            long_score = float(scores.get(long_kw, 0.0))
            if long_score + 1e-9 < short_score * score_ratio:
                continue
            if short_kw in long_kw or long_units[: len(short_units)] == short_units:
                promotions.append((long_score, long_kw))

    for _, kw in sorted(promotions, key=lambda item: item[0], reverse=True)[:max_extra]:
        if kw in selected_set:
            continue
        if len(selected) < max_k:
            selected.append(kw)
            selected_set.add(kw)
            continue
        replace_idx = min(range(len(selected)), key=lambda i: float(scores.get(selected[i], 0.0)))
        selected_set.discard(selected[replace_idx])
        selected[replace_idx] = kw
        selected_set.add(kw)

    selected.sort(key=lambda kw: float(scores.get(kw, 0.0)), reverse=True)
    return selected[:max_k]


def _select_prompt_keywords(ranked: Sequence[str], scores: Dict[str, float], args) -> List[str]:
    if len(ranked) == 0:
        return []
    top_score = float(scores.get(ranked[0], 0.0))
    keep_threshold = max(float(args.prompt_score_threshold), top_score * float(args.prompt_relative_threshold))
    selected = [kw for kw in ranked if float(scores.get(kw, 0.0)) >= keep_threshold]
    if len(selected) == 0:
        selected = [ranked[0]]
    return _promote_nested_keywords(
        ranked,
        selected,
        scores,
        max_k=max(1, int(args.prompt_max_injected_keywords)),
        score_ratio=float(args.nested_keyword_promotion_score_ratio),
        min_long_chars=max(2, int(args.nested_keyword_promotion_min_long_chars)),
        max_extra=max(0, int(args.nested_keyword_promotion_max_extra)),
    )


def _select_rescore_keywords(ranked: Sequence[str], scores: Dict[str, float], args) -> List[str]:
    max_k = max(1, int(args.rescore_max_keywords))
    return _promote_nested_keywords(
        ranked,
        list(ranked)[:max_k],
        scores,
        max_k=max_k,
        score_ratio=float(args.nested_keyword_promotion_score_ratio),
        min_long_chars=max(2, int(args.nested_keyword_promotion_min_long_chars)),
        max_extra=max(0, int(args.nested_keyword_promotion_max_extra)),
    )


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose KWS top-k and CB-SenseVoice prompt/rescore hotword coverage.")
    parser.add_argument("--root", default="../datasets/aishell/data_aishell")
    parser.add_argument("--split", default="test")
    parser.add_argument("--kw-type", default="tts")
    parser.add_argument("--features-size", nargs=2, type=int, default=[150, 750])
    parser.add_argument("--hotwords-per-group", type=int, default=100)
    parser.add_argument(
        "--kws-ckpt",
        default="/root/autodl-tmp/src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--output-csv", default="logs/kws_topk_recall_aishell.csv")
    parser.add_argument("--limit", type=int)

    parser.add_argument("--kws-positive-threshold", type=float, default=0.30)
    parser.add_argument("--kws-topk-per-group", type=int, default=6)
    parser.add_argument("--kws-max-prompt-keywords", type=int, default=28)
    parser.add_argument("--prompt-max-injected-keywords", type=int, default=4)
    parser.add_argument("--prompt-score-threshold", type=float, default=0.55)
    parser.add_argument("--prompt-relative-threshold", type=float, default=0.80)
    parser.add_argument("--rescore-max-keywords", type=int, default=12)
    parser.add_argument("--nested-keyword-promotion-score-ratio", type=float, default=0.95)
    parser.add_argument("--nested-keyword-promotion-min-long-chars", type=int, default=3)
    parser.add_argument("--nested-keyword-promotion-max-extra", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = AishellHotwordDataset(
        root=os.path.join(args.root, "hotword"),
        split=args.split,
        size=tuple(args.features_size),
        r1_only=False,
        hotwords_per_group=int(args.hotwords_per_group),
        kw_type=args.kw_type,
        load_audio=False,
    )
    model = KWSModel.load_from_checkpoint(args.kws_ckpt, map_location=device).to(device).eval()

    topk_values = sorted(set(int(k) for k in args.topk))
    totals = {
        "samples": 0,
        "samples_with_true": 0,
        "true_hotwords": 0,
        "prompt_hits": 0,
        "rescore_hits": 0,
        "cb_pool_hits": 0,
    }
    topk_hits = {k: 0 for k in topk_values}
    sample_any_hits = {k: 0 for k in topk_values}
    rows = []

    with torch.inference_mode():
        sample_count = len(dataset) if args.limit is None else min(len(dataset), max(0, args.limit))
        for idx in tqdm(range(sample_count), desc="KWS top-k"):
            item = dataset[idx]
            all_records = []
            true_keywords = []

            for group_idx, (features, labels, mask, group) in enumerate(
                zip(item["features"], item["hotword_labels"], item["hotword_mask"], dataset.database)
            ):
                features = features.to(device)
                probs_parts = []
                for chunk in torch.split(features, max(1, int(args.chunk_size)), dim=0):
                    logits = model(input_features=chunk).logits
                    probs_parts.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu())
                probs = torch.cat(probs_parts, dim=0)
                labels = labels.cpu().long()
                mask = mask.cpu().long()
                keywords = list(group["keywords"])

                for local_idx, (kw, score, label, valid) in enumerate(zip(keywords, probs.tolist(), labels.tolist(), mask.tolist())):
                    score = float(score) * float(valid)
                    record = {
                        "keyword": str(kw),
                        "score": score,
                        "label": int(label),
                        "group_idx": int(group_idx),
                        "local_idx": int(local_idx),
                    }
                    all_records.append(record)
                    if int(label) == 1:
                        true_keywords.append(str(kw))

            if len(true_keywords) == 0:
                continue

            totals["samples"] += 1
            totals["samples_with_true"] += 1
            totals["true_hotwords"] += len(true_keywords)
            true_set = set(true_keywords)
            ranked_records = sorted(all_records, key=lambda rec: rec["score"], reverse=True)
            ranked_keywords = [rec["keyword"] for rec in ranked_records]
            score_map = {rec["keyword"]: float(rec["score"]) for rec in ranked_records}

            for k in topk_values:
                top_set = set(ranked_keywords[:k])
                hit = len(true_set & top_set)
                topk_hits[k] += hit
                sample_any_hits[k] += int(hit > 0)

            cb_keywords = []
            cb_scores: Dict[str, float] = {}
            for group_idx in sorted(set(rec["group_idx"] for rec in all_records)):
                group_records = [rec for rec in all_records if rec["group_idx"] == group_idx]
                group_records.sort(key=lambda rec: rec["score"], reverse=True)
                group_top = group_records[: max(1, int(args.kws_topk_per_group))]
                selected = [rec for rec in group_top if rec["score"] >= float(args.kws_positive_threshold)]
                if len(selected) == 0 and len(group_top) > 0:
                    selected = group_top[:1]
                for rec in selected:
                    kw = rec["keyword"]
                    cb_keywords.append(kw)
                    cb_scores[kw] = max(float(cb_scores.get(kw, 0.0)), float(rec["score"]))

            cb_ranked = _sort_keywords_by_score(cb_keywords, cb_scores)[: max(0, int(args.kws_max_prompt_keywords))]
            cb_scores = {kw: float(cb_scores.get(kw, 0.0)) for kw in cb_ranked}
            prompt_keywords = _select_prompt_keywords(cb_ranked, cb_scores, args)
            rescore_keywords = _select_rescore_keywords(cb_ranked, cb_scores, args)

            prompt_hit = len(true_set & set(prompt_keywords))
            rescore_hit = len(true_set & set(rescore_keywords))
            cb_pool_hit = len(true_set & set(cb_ranked))
            totals["prompt_hits"] += prompt_hit
            totals["rescore_hits"] += rescore_hit
            totals["cb_pool_hits"] += cb_pool_hit

            true_ranks = [
                ranked_keywords.index(kw) + 1
                for kw in true_keywords
                if kw in ranked_keywords
            ]
            rows.append(
                {
                    "idx": idx,
                    "transcript": item["transcript"],
                    "true_keywords": "|".join(true_keywords),
                    "num_true": len(true_keywords),
                    "best_true_rank": min(true_ranks) if len(true_ranks) else -1,
                    "worst_true_rank": max(true_ranks) if len(true_ranks) else -1,
                    **{f"recall_at_{k}": _safe_div(len(true_set & set(ranked_keywords[:k])), len(true_set)) for k in topk_values},
                    "cb_pool_recall": _safe_div(cb_pool_hit, len(true_set)),
                    "prompt_recall": _safe_div(prompt_hit, len(true_set)),
                    "rescore_recall": _safe_div(rescore_hit, len(true_set)),
                    "cb_pool_size": len(cb_ranked),
                    "prompt_size": len(prompt_keywords),
                    "rescore_size": len(rescore_keywords),
                    "top12_keywords": "|".join(ranked_keywords[:12]),
                    "prompt_keywords": "|".join(prompt_keywords),
                    "rescore_keywords": "|".join(rescore_keywords),
                }
            )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": totals["samples_with_true"],
        "true_hotwords": totals["true_hotwords"],
        **{f"micro_recall@{k}": _safe_div(topk_hits[k], totals["true_hotwords"]) for k in topk_values},
        **{f"sample_any_recall@{k}": _safe_div(sample_any_hits[k], totals["samples_with_true"]) for k in topk_values},
        "cb_pool_micro_recall": _safe_div(totals["cb_pool_hits"], totals["true_hotwords"]),
        "prompt_micro_recall": _safe_div(totals["prompt_hits"], totals["true_hotwords"]),
        "rescore_micro_recall": _safe_div(totals["rescore_hits"], totals["true_hotwords"]),
        "output_csv": args.output_csv,
    }
    print(summary)


if __name__ == "__main__":
    main()
