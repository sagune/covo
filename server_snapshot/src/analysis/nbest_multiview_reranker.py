import argparse
import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from nbest_consensus import edit_distance, normalize_surface, read_detail


def as_float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def as_int(row: Dict[str, object], key: str, default: int = 0) -> int:
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


def parse_views(spec: str) -> List[Tuple[str, str]]:
    views = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"view spec must be name=path, got {item}")
        name, path = item.split("=", 1)
        views.append((name.strip(), path.strip()))
    if not views:
        raise ValueError("empty view spec")
    return views


def load_multiview(spec: str) -> Dict[str, List[Dict[str, object]]]:
    out = defaultdict(list)
    for view_id, (view_name, path) in enumerate(parse_views(spec)):
        groups = read_detail(path)
        for idx, rows in groups.items():
            for row in rows:
                item = dict(row)
                item["view"] = view_name
                item["view_id"] = view_id
                item["view_rank"] = as_int(row, "rank", 999999)
                item["rank"] = len(out[idx]) + 1
                out[idx].append(item)
    for rows in out.values():
        rows.sort(key=lambda r: (as_int(r, "view_id"), as_int(r, "view_rank")))
        seen = {}
        deduped = []
        for row in rows:
            key = normalize_surface(str(row.get("pred", "")))
            old = seen.get(key)
            if old is None:
                seen[key] = row
                deduped.append(row)
                continue
            if as_float(row, "cer", 999.0) < as_float(old, "cer", 999.0):
                old.update(row)
        rows[:] = deduped
    return dict(out)


def target_index(rows: List[Dict[str, object]], label_mode: str) -> int:
    if label_mode == "oracle":
        return min(
            range(len(rows)),
            key=lambda i: (
                -as_float(rows[i], "entity_recall", 0.0),
                as_float(rows[i], "hotword_only_cer", 0.0),
                as_float(rows[i], "cer", 999.0),
                as_int(rows[i], "view_rank", 999999),
            ),
        )
    if label_mode == "cer":
        return min(range(len(rows)), key=lambda i: (as_float(rows[i], "cer", 999.0), as_int(rows[i], "view_rank", 999999)))
    if label_mode == "cer_recall":
        return min(
            range(len(rows)),
            key=lambda i: (
                as_float(rows[i], "cer", 999.0)
                + 0.15 * as_float(rows[i], "hotword_only_cer", 0.0)
                - 0.04 * as_float(rows[i], "entity_recall", 0.0),
                as_int(rows[i], "view_rank", 999999),
            ),
        )
    raise ValueError(f"unsupported label mode: {label_mode}")


def features_for_rows(rows: List[Dict[str, object]], label_mode: str, num_views: int):
    total_vals = [as_float(r, "total_score") for r in rows]
    asr_vals = [as_float(r, "asr_score") for r in rows]
    exact_vals = [as_float(r, "exact_score") for r in rows]
    phon_vals = [as_float(r, "phonetic_score") for r in rows]
    cons_vals = [as_float(r, "consensus_score") for r in rows]
    support_vals = [as_float(r, "consensus_support") for r in rows]
    pred_lens = [len(normalize_surface(str(r.get("pred", "")))) for r in rows]
    top_by_view = {}
    for row in rows:
        view_id = as_int(row, "view_id")
        if view_id not in top_by_view or as_int(row, "view_rank", 999999) < as_int(top_by_view[view_id], "view_rank", 999999):
            top_by_view[view_id] = row
    primary_top = top_by_view.get(0, rows[0]).get("pred", "")
    ref_len_proxy = pred_lens[0] if pred_lens else 1
    nbest = max(len(rows), 1)
    feats = []
    for row in rows:
        view_id = as_int(row, "view_id")
        view_rank = as_int(row, "view_rank", 1)
        pred = str(row.get("pred", ""))
        norm_len = len(normalize_surface(pred))
        len_rel = (norm_len - ref_len_proxy) / max(ref_len_proxy, 1)
        d_primary = norm_dist(pred, str(primary_top))
        d_view_top = norm_dist(pred, str(top_by_view.get(view_id, rows[0]).get("pred", "")))
        view_bits = [1.0 if view_id == i else 0.0 for i in range(num_views)]
        feats.append([
            1.0,
            -((view_rank - 1) / 15.0),
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
            -d_primary,
            -d_view_top,
            nbest / 24.0,
            norm_len / 40.0,
        ] + view_bits)
    return torch.tensor(feats, dtype=torch.float32), target_index(rows, label_mode)


def make_examples(groups: Dict[str, List[Dict[str, object]]], label_mode: str, num_views: int):
    return [(idx, rows, *features_for_rows(rows, label_mode, num_views)) for idx, rows in groups.items()]


class LinearRanker(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x @ self.weight


class MLPRanker(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(examples, epochs: int, lr: float, weight_decay: float, seed: int, model_type: str):
    random.seed(seed)
    torch.manual_seed(seed)
    if model_type == "linear":
        model = LinearRanker(examples[0][2].shape[1])
    elif model_type == "mlp":
        model = MLPRanker(examples[0][2].shape[1])
    else:
        raise ValueError(f"unsupported model type: {model_type}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        random.shuffle(examples)
        loss_sum = 0.0
        for _, _, x, y in examples:
            scores = model(x)
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([y]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        if epoch in {0, epochs - 1} or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={loss_sum / max(len(examples), 1):.4f}")
    return model


def summarize(rows: List[Dict[str, object]], selector: str, picked: List[Dict[str, object]]):
    n = len(picked)
    return {
        "selector": selector,
        "samples": n,
        "entity_recall": sum(as_float(r, "entity_recall") for r in picked) / max(n, 1),
        "cer": sum(as_float(r, "cer") for r in picked) / max(n, 1),
        "hotword_only_cer": sum(as_float(r, "hotword_only_cer") for r in picked) / max(n, 1),
        "primary_top1_rate": sum(1 for r in picked if as_int(r, "view_id") == 0 and as_int(r, "view_rank") == 1) / max(n, 1),
        "view0_rate": sum(1 for r in picked if as_int(r, "view_id") == 0) / max(n, 1),
        "view1_rate": sum(1 for r in picked if as_int(r, "view_id") == 1) / max(n, 1),
    }


def evaluate(model, examples, name: str):
    picked = []
    with torch.no_grad():
        for _, rows, x, _ in examples:
            picked.append(rows[int(torch.argmax(model(x)).item())])
    return summarize([], name, picked), picked


def oracle_summary(examples, label_mode: str):
    picked = []
    for _, rows, _, _ in examples:
        picked.append(rows[target_index(rows, label_mode)])
    return summarize([], f"oracle_{label_mode}", picked)


def primary_top1_summary(examples):
    picked = []
    for _, rows, _, _ in examples:
        primary = [r for r in rows if as_int(r, "view_id") == 0 and as_int(r, "view_rank") == 1]
        picked.append(primary[0] if primary else rows[0])
    return summarize([], "primary_top1", picked)


def write_csv(path: str, rows: List[Dict[str, object]]):
    fields = ["selector", "samples", "entity_recall", "cer", "hotword_only_cer", "primary_top1_rate", "view0_rate", "view1_rate"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-views", required=True, help="name=detail.csv,name2=detail.csv")
    parser.add_argument("--test-views", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--label-mode", choices=["oracle", "cer", "cer_recall"], default="cer_recall")
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--holdout-mod", type=int, default=0)
    args = parser.parse_args()

    num_views = len(parse_views(args.test_views))
    train_groups = load_multiview(args.train_views)
    test_groups = load_multiview(args.test_views)
    if args.train_views == args.test_views and args.holdout_mod > 1:
        test_groups = {k: v for k, v in test_groups.items() if int(k) % args.holdout_mod == 0}
        train_groups = {k: v for k, v in train_groups.items() if int(k) % args.holdout_mod != 0}
        name = f"multiview_{args.model_type}_holdout_mod{args.holdout_mod}_{args.label_mode}"
    elif args.train_views == args.test_views:
        name = f"multiview_{args.model_type}_insample_{args.label_mode}"
    else:
        name = f"multiview_{args.model_type}_{args.label_mode}"

    train_examples = make_examples(train_groups, args.label_mode, num_views)
    test_examples = make_examples(test_groups, args.label_mode, num_views)
    model = train(train_examples, args.epochs, args.lr, args.weight_decay, args.seed, args.model_type)
    result, _ = evaluate(model, test_examples, name)
    rows = [
        primary_top1_summary(test_examples),
        oracle_summary(test_examples, "cer"),
        oracle_summary(test_examples, "cer_recall"),
        oracle_summary(test_examples, "oracle"),
        result,
    ]
    write_csv(args.output_csv, rows)
    print(rows)
    if hasattr(model, "weight"):
        print("weights", [round(float(x), 4) for x in model.weight.detach().tolist()])


if __name__ == "__main__":
    main()
