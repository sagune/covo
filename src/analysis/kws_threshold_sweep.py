import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from data.dataset import AishellHotwordDataset
from model.model import KWSModel


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _metrics(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> Dict[str, float]:
    preds = scores >= float(threshold)
    labels = labels.bool()
    tp = int((preds & labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_positive": tp + fp,
    }


def _collect_scores(args) -> Tuple[torch.Tensor, torch.Tensor]:
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

    scores: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    with torch.inference_mode():
        for item in tqdm(dataset, desc="KWS threshold sweep"):
            for features, hotword_labels, mask in zip(
                item["features"], item["hotword_labels"], item["hotword_mask"]
            ):
                parts = []
                for chunk in torch.split(features.to(device), max(1, int(args.chunk_size)), dim=0):
                    logits = model(input_features=chunk).logits
                    parts.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu())
                group_scores = torch.cat(parts, dim=0)
                group_mask = mask.detach().cpu().float()
                scores.append(group_scores * group_mask)
                labels.append(hotword_labels.detach().cpu().long() * group_mask.long())

    return torch.cat(labels, dim=0).long(), torch.cat(scores, dim=0).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep a fixed KWS decision threshold on a hotword test set.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--kw-type", default="natural")
    parser.add_argument("--features-size", nargs=2, type=int, default=[150, 750])
    parser.add_argument("--hotwords-per-group", type=int, default=100)
    parser.add_argument("--kws-ckpt", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--target-recall", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--output-csv", default="logs/kws_threshold_sweep.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    labels, scores = _collect_scores(args)
    thresholds = torch.linspace(0.0, 1.0, steps=max(11, int(args.grid_size))).tolist()
    rows = [_metrics(labels, scores, threshold) for threshold in thresholds]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_f1 = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"]))
    summary = {
        "total_candidates": int(labels.numel()),
        "total_true": int(labels.sum().item()),
        "best_f1": best_f1,
    }
    for target in args.target_recall:
        feasible = [row for row in rows if row["recall"] >= float(target)]
        summary[f"max_precision_at_recall>={target}"] = (
            max(feasible, key=lambda row: (row["precision"], row["f1"], row["threshold"])) if feasible else None
        )
    print(summary)
    print("output_csv:", args.output_csv)


if __name__ == "__main__":
    main()
