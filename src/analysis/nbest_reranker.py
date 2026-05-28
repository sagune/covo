import argparse
import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from nbest_consensus import edit_distance, normalize_surface, read_detail


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def zscore(values: List[float], value: float) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (value - mean) / ((var ** 0.5) + 1e-6)


def norm_dist(a: str, b: str) -> float:
    a = normalize_surface(a)
    b = normalize_surface(b)
    return edit_distance(a, b) / max(len(a), len(b), 1)


def _target_index(rows: List[Dict[str, str]], label_mode: str) -> int:
    if label_mode == "oracle":
        for i, row in enumerate(rows):
            if str(row.get("is_oracle", "")).lower() == "true":
                return i
        return 0
    if label_mode == "cer":
        return min(
            range(len(rows)),
            key=lambda i: (as_float(rows[i], "cer", 999.0), as_int(rows[i], "rank", i + 1)),
        )
    if label_mode == "cer_recall":
        return min(
            range(len(rows)),
            key=lambda i: (
                as_float(rows[i], "cer", 999.0)
                + 0.2 * as_float(rows[i], "hotword_only_cer", 0.0)
                - 0.05 * as_float(rows[i], "entity_recall", 0.0),
                as_int(rows[i], "rank", i + 1),
            ),
        )
    raise ValueError(f"unsupported label mode: {label_mode}")


def features_for_rows(rows: List[Dict[str, str]], label_mode: str) -> Tuple[torch.Tensor, int]:
    total_vals = [as_float(r, "total_score") for r in rows]
    asr_vals = [as_float(r, "asr_score") for r in rows]
    exact_vals = [as_float(r, "exact_score") for r in rows]
    phon_vals = [as_float(r, "phonetic_score") for r in rows]
    cons_vals = [as_float(r, "consensus_score") for r in rows]
    support_vals = [as_float(r, "consensus_support") for r in rows]
    pred_lens = [len(normalize_surface(r.get("pred", ""))) for r in rows]
    ref_len_proxy = pred_lens[0] if pred_lens else 1
    nbest = max(len(rows), 1)
    top_pred = rows[0].get("pred", "") if rows else ""

    feats = []
    for i, row in enumerate(rows):
        rank = as_int(row, "rank", i + 1)
        pred = row.get("pred", "")
        rank_norm = (rank - 1) / max(nbest - 1, 1)
        len_rel = (len(normalize_surface(pred)) - ref_len_proxy) / max(ref_len_proxy, 1)
        d_top = norm_dist(pred, top_pred)
        feats.append([
            1.0,
            -rank_norm,
            zscore(total_vals, as_float(row, "total_score")),
            zscore(asr_vals, as_float(row, "asr_score")),
            zscore(exact_vals, as_float(row, "exact_score")),
            zscore(phon_vals, as_float(row, "phonetic_score")),
            zscore(cons_vals, as_float(row, "consensus_score")),
            zscore(support_vals, as_float(row, "consensus_support")),
            as_float(row, "exact_score"),
            as_float(row, "phonetic_score"),
            as_float(row, "consensus_score"),
            as_float(row, "consensus_support") / 10.0,
            len_rel,
            abs(len_rel),
            -d_top,
            nbest / 16.0,
        ])
    return torch.tensor(feats, dtype=torch.float32), _target_index(rows, label_mode)


def make_examples(groups: Dict[str, List[Dict[str, str]]], label_mode: str):
    examples = []
    for idx, rows in groups.items():
        x, y = features_for_rows(rows, label_mode)
        examples.append((idx, rows, x, y))
    return examples


class LinearRanker(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def train_ranker(examples, epochs: int, lr: float, weight_decay: float, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    dim = examples[0][2].shape[1]
    model = LinearRanker(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        random.shuffle(examples)
        total = 0.0
        for _, _, x, y in examples:
            score = model(x)
            loss = torch.nn.functional.cross_entropy(score.unsqueeze(0), torch.tensor([y]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if epoch in {0, epochs - 1} or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={total / max(len(examples), 1):.4f}")
    return model


def evaluate(model, examples, name: str):
    n = 0
    sums = defaultdict(float)
    top1 = 0
    oracle = 0
    selected_rows = []
    with torch.no_grad():
        for idx, rows, x, _ in examples:
            scores = model(x)
            pick = int(torch.argmax(scores).item())
            row = rows[pick]
            n += 1
            sums["entity_recall"] += as_float(row, "entity_recall")
            sums["cer"] += as_float(row, "cer")
            sums["hotword_only_cer"] += as_float(row, "hotword_only_cer")
            if str(row.get("is_top1", "")).lower() == "true":
                top1 += 1
            if str(row.get("is_oracle", "")).lower() == "true":
                oracle += 1
            selected_rows.append({
                "idx": idx,
                "rank": row.get("rank", ""),
                "pred": row.get("pred", ""),
                "ref": row.get("ref", ""),
                "cer": row.get("cer", ""),
                "entity_recall": row.get("entity_recall", ""),
                "hotword_only_cer": row.get("hotword_only_cer", ""),
            })
    return {
        "selector": name,
        "samples": n,
        "entity_recall": sums["entity_recall"] / max(n, 1),
        "cer": sums["cer"] / max(n, 1),
        "hotword_only_cer": sums["hotword_only_cer"] / max(n, 1),
        "top1_rate": top1 / max(n, 1),
        "oracle_rate": oracle / max(n, 1),
    }, selected_rows


def baseline_top1(examples):
    n = 0
    sums = defaultdict(float)
    for _, rows, _, _ in examples:
        row = rows[0]
        n += 1
        sums["entity_recall"] += as_float(row, "entity_recall")
        sums["cer"] += as_float(row, "cer")
        sums["hotword_only_cer"] += as_float(row, "hotword_only_cer")
    return {
        "selector": "top1",
        "samples": n,
        "entity_recall": sums["entity_recall"] / max(n, 1),
        "cer": sums["cer"] / max(n, 1),
        "hotword_only_cer": sums["hotword_only_cer"] / max(n, 1),
        "top1_rate": 1.0,
        "oracle_rate": 0.0,
    }


def write_rows(path: str, rows: List[Dict[str, object]]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selector", "samples", "entity_recall", "cer", "hotword_only_cer", "top1_rate", "oracle_rate"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-detail", required=True)
    parser.add_argument("--test-detail", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pred-csv", default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--label-mode", choices=["oracle", "cer", "cer_recall"], default="oracle")
    parser.add_argument("--holdout-mod", type=int, default=0, help="If train and test are the same file, evaluate idx % holdout_mod == 0.")
    args = parser.parse_args()

    train_groups = read_detail(args.train_detail)
    test_groups = read_detail(args.test_detail)
    if args.train_detail == args.test_detail and args.holdout_mod > 1:
        held = {k: v for k, v in test_groups.items() if int(k) % args.holdout_mod == 0}
        train_groups = {k: v for k, v in train_groups.items() if int(k) % args.holdout_mod != 0}
        test_groups = held
        eval_name = f"linear_reranker_holdout_mod{args.holdout_mod}"
    elif args.train_detail == args.test_detail:
        eval_name = "linear_reranker_in_sample"
    else:
        eval_name = "linear_reranker"

    train_examples = make_examples(train_groups, args.label_mode)
    test_examples = make_examples(test_groups, args.label_mode)
    model = train_ranker(train_examples, args.epochs, args.lr, args.weight_decay, args.seed)

    result, selected = evaluate(model, test_examples, eval_name)
    rows = [baseline_top1(test_examples), result]
    write_rows(args.output_csv, rows)
    if args.pred_csv:
        with open(args.pred_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "rank", "cer", "entity_recall", "hotword_only_cer", "ref", "pred"])
            writer.writeheader()
            writer.writerows(selected)
    print(rows)
    print("weights", [round(float(x), 4) for x in model.weight.detach().tolist()])


if __name__ == "__main__":
    main()
