import argparse
import csv
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"[·•‧・]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def edit_distance(a: Iterable[str], b: Iterable[str]) -> int:
    a = list(a)
    b = list(b)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def cer(ref: str, pred: str) -> float:
    ref = normalize_surface(ref)
    pred = normalize_surface(pred)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(ref, pred) / len(ref)


def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def read_detail(path: str) -> Dict[str, List[Dict[str, str]]]:
    groups = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("idx")
            if idx is None:
                idx = row.get("\ufeffidx")
            row["idx"] = str(idx)
            groups[str(idx)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: _as_int(r, "rank", 999999))
    return dict(groups)


@lru_cache(maxsize=200000)
def norm_distance(a: str, b: str) -> float:
    a = normalize_surface(a)
    b = normalize_surface(b)
    denom = max(len(a), len(b), 1)
    return edit_distance(a, b) / denom


def softmax_weights(rows: List[Dict[str, str]], key: str, temperature: float) -> List[float]:
    vals = [_as_float(r, key, 0.0) / max(temperature, 1e-6) for r in rows]
    m = max(vals) if vals else 0.0
    exps = [math.exp(max(min(v - m, 50.0), -50.0)) for v in vals]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def rank_weights(rows: List[Dict[str, str]], decay: float) -> List[float]:
    weights = [decay ** max(_as_int(r, "rank", 1) - 1, 0) for r in rows]
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def combine_weights(weight_sets: List[List[float]]) -> List[float]:
    if not weight_sets:
        return []
    out = [1.0] * len(weight_sets[0])
    for weights in weight_sets:
        for i, w in enumerate(weights):
            out[i] *= max(w, 1e-8)
    total = sum(out) or 1.0
    return [w / total for w in out]


def select_mbr(
    rows: List[Dict[str, str]],
    alpha: float,
    weights: List[float],
    keyword_bonus: float = 0.0,
) -> Dict[str, str]:
    preds = [r.get("pred", "") for r in rows]
    if not weights:
        weights = [1.0 / max(len(rows), 1)] * len(rows)
    best_row = rows[0]
    best_score = float("inf")
    for i, row in enumerate(rows):
        avg_dist = 0.0
        for j, other in enumerate(rows):
            avg_dist += weights[j] * norm_distance(preds[i], preds[j])
        rank_penalty = alpha * max(_as_int(row, "rank", 1) - 1, 0)
        kw_reward = keyword_bonus * (
            _as_float(row, "exact_score", 0.0)
            + 0.5 * _as_float(row, "consensus_score", 0.0)
            + 0.1 * _as_float(row, "consensus_support", 0.0)
        )
        score = avg_dist + rank_penalty - kw_reward
        if score < best_score:
            best_score = score
            best_row = row
    return best_row


def align_base_to_candidate(base: str, cand: str) -> Tuple[List[str], List[str]]:
    base = list(base)
    cand = list(cand)
    m, n = len(base), len(cand)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        bt[i][0] = "up"
    for j in range(1, n + 1):
        dp[0][j] = j
        bt[0][j] = "left"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (0 if base[i - 1] == cand[j - 1] else 1)
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = min(sub, delete, insert)
            dp[i][j] = best
            bt[i][j] = "diag" if best == sub else ("up" if best == delete else "left")
    i, j = m, n
    a_base, a_cand = [], []
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "diag":
            a_base.append(base[i - 1])
            a_cand.append(cand[j - 1])
            i -= 1
            j -= 1
        elif step == "up":
            a_base.append(base[i - 1])
            a_cand.append("")
            i -= 1
        else:
            a_base.append("")
            a_cand.append(cand[j - 1])
            j -= 1
    return list(reversed(a_base)), list(reversed(a_cand))


def consensus_decode(rows: List[Dict[str, str]], weights: List[float]) -> str:
    if not rows:
        return ""
    base = normalize_surface(rows[0].get("pred", ""))
    if not base:
        return normalize_surface(rows[0].get("pred", ""))
    if not weights:
        weights = [1.0 / len(rows)] * len(rows)

    slot_votes = [defaultdict(float) for _ in range(len(base))]
    insertion_votes = [defaultdict(float) for _ in range(len(base) + 1)]
    for row, weight in zip(rows, weights):
        cand = normalize_surface(row.get("pred", ""))
        a_base, a_cand = align_base_to_candidate(base, cand)
        pos = 0
        insert_buffers = defaultdict(list)
        for b, c in zip(a_base, a_cand):
            if b == "":
                insert_buffers[pos].append(c)
                continue
            if insert_buffers[pos]:
                for k, ch in enumerate(insert_buffers[pos]):
                    insertion_votes[pos][(k, ch)] += weight
                insert_buffers[pos].clear()
            slot_votes[pos][c] += weight
            pos += 1
        if insert_buffers[pos]:
            for k, ch in enumerate(insert_buffers[pos]):
                insertion_votes[pos][(k, ch)] += weight

    pieces = []
    for pos in range(len(base) + 1):
        by_k = defaultdict(lambda: defaultdict(float))
        for (k, ch), vote in insertion_votes[pos].items():
            by_k[k][ch] += vote
        for k in sorted(by_k):
            ch, vote = max(by_k[k].items(), key=lambda kv: kv[1])
            if ch and vote > 0.5:
                pieces.append(ch)
        if pos < len(base):
            ch, vote = max(slot_votes[pos].items(), key=lambda kv: kv[1]) if slot_votes[pos] else (base[pos], 0.0)
            if ch and vote >= 0.35:
                pieces.append(ch)
    return "".join(pieces)


def evaluate_existing(groups: Dict[str, List[Dict[str, str]]], selector_name: str, selector):
    n = 0
    sums = defaultdict(float)
    top1 = 0
    oracle = 0
    for rows in groups.values():
        row = selector(rows)
        n += 1
        sums["entity_recall"] += _as_float(row, "entity_recall")
        sums["cer"] += _as_float(row, "cer")
        sums["hotword_only_cer"] += _as_float(row, "hotword_only_cer")
        if str(row.get("is_top1", "")).lower() == "true":
            top1 += 1
        if str(row.get("is_oracle", "")).lower() == "true":
            oracle += 1
    return {
        "selector": selector_name,
        "samples": n,
        "entity_recall": sums["entity_recall"] / max(n, 1),
        "cer": sums["cer"] / max(n, 1),
        "hotword_only_cer": sums["hotword_only_cer"] / max(n, 1),
        "top1_rate": top1 / max(n, 1),
        "oracle_rate": oracle / max(n, 1),
    }


def evaluate_consensus(groups: Dict[str, List[Dict[str, str]]], name: str, weight_fn):
    n = 0
    total_cer = 0.0
    changed = 0
    for rows in groups.values():
        pred = consensus_decode(rows, weight_fn(rows))
        ref = rows[0].get("ref", "")
        total_cer += cer(ref, pred)
        n += 1
        if pred != normalize_surface(rows[0].get("pred", "")):
            changed += 1
    return {
        "selector": name,
        "samples": n,
        "entity_recall": "",
        "cer": total_cer / max(n, 1),
        "hotword_only_cer": "",
        "top1_rate": 1.0 - changed / max(n, 1),
        "oracle_rate": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--keyword-bonuses", default="0,0.01,0.03")
    args = parser.parse_args()

    groups = read_detail(args.detail_csv)
    results = []
    results.append(evaluate_existing(groups, "top1", lambda rows: rows[0]))
    results.append(evaluate_existing(groups, "oracle", lambda rows: next((r for r in rows if str(r.get("is_oracle", "")).lower() == "true"), rows[0])))

    for alpha in [float(x) for x in args.alphas.split(",") if x.strip()]:
        for bonus in [float(x) for x in args.keyword_bonuses.split(",") if x.strip()]:
            results.append(evaluate_existing(
                groups,
                f"mbr_uniform_a{alpha:g}_kw{bonus:g}",
                lambda rows, a=alpha, b=bonus: select_mbr(rows, a, [1.0 / len(rows)] * len(rows), b),
            ))
            results.append(evaluate_existing(
                groups,
                f"mbr_rank_a{alpha:g}_kw{bonus:g}",
                lambda rows, a=alpha, b=bonus: select_mbr(rows, a, rank_weights(rows, 0.55), b),
            ))
            results.append(evaluate_existing(
                groups,
                f"mbr_score_a{alpha:g}_kw{bonus:g}",
                lambda rows, a=alpha, b=bonus: select_mbr(rows, a, softmax_weights(rows, "total_score", 1.0), b),
            ))

    results.append(evaluate_consensus(groups, "consensus_rank", lambda rows: rank_weights(rows, 0.55)))
    results.append(evaluate_consensus(groups, "consensus_score", lambda rows: softmax_weights(rows, "total_score", 1.0)))
    results.append(evaluate_consensus(groups, "consensus_rank_score", lambda rows: combine_weights([rank_weights(rows, 0.55), softmax_weights(rows, "total_score", 1.0)])))

    results.sort(key=lambda r: float(r["cer"]))
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selector", "samples", "entity_recall", "cer", "hotword_only_cer", "top1_rate", "oracle_rate"])
        writer.writeheader()
        writer.writerows(results)

    for row in results[:12]:
        print(row)


if __name__ == "__main__":
    main()
