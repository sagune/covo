import argparse
import os
import sys
from collections import defaultdict


def read_lines(path: str):
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return [line.rstrip("\n") for line in f]
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", required=True, help="path to positives.tsv")
    parser.add_argument("--output", required=True, help="path to write rebuilt keywords.txt")
    parser.add_argument("--strict", action="store_true", help="fail on conflicts")
    args = parser.parse_args()

    lines = read_lines(args.positives)
    idx_to_kw = {}
    conflicts = defaultdict(set)
    max_idx = -1

    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        # parts: code, kw1, idx1, rev1, kw2, idx2, rev2, ...
        for i in range(1, len(parts) - 2, 3):
            kw = parts[i]
            idx_str = parts[i + 1]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
            if idx in idx_to_kw and idx_to_kw[idx] != kw:
                conflicts[idx].update({idx_to_kw[idx], kw})
            else:
                idx_to_kw[idx] = kw

    if conflicts:
        conflict_lines = []
        for idx in sorted(conflicts.keys()):
            kws = sorted(conflicts[idx])
            conflict_lines.append(f"idx={idx}\t" + ",".join(kws))
        msg = "Conflicting keywords for the same idx:\n" + "\n".join(conflict_lines[:20])
        if args.strict:
            raise RuntimeError(msg)
        else:
            print(msg, file=sys.stderr)

    if max_idx < 0:
        raise RuntimeError("No valid idx found in positives.tsv")

    keywords = [""] * (max_idx + 1)
    missing = []
    for idx in range(max_idx + 1):
        if idx in idx_to_kw:
            keywords[idx] = idx_to_kw[idx]
        else:
            missing.append(idx)
            keywords[idx] = ""

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for kw in keywords:
            f.write(f"{kw}\n")

    if missing:
        print(f"Missing keywords for {len(missing)} indices. First 20: {missing[:20]}", file=sys.stderr)


if __name__ == "__main__":
    main()
