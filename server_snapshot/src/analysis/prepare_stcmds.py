#!/usr/bin/env python3
"""Build a deterministic stratified ST-CMDS evaluation manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    rows = []
    for txt in sorted(args.root.glob("*.txt")):
        wav = txt.with_suffix(".wav")
        if not wav.exists():
            continue
        reference = txt.read_text(encoding="utf-8-sig").strip()
        if not reference:
            continue
        name = wav.stem
        speaker = name.split("P", 1)[-1].split("A", 1)[0].split("I", 1)[0]
        device = "A" if "A" in name.split("P", 1)[-1] else "I"
        rows.append({"id": name, "reference": reference, "wav": str(wav),
                     "speaker": speaker, "device": device, "source": "ST-CMDS"})

    rng = random.Random(args.seed)
    selected = []
    groups = {}
    for row in rows:
        groups.setdefault(row["device"], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    if args.limit >= len(rows):
        selected = rows
    else:
        for device, group in sorted(groups.items()):
            count = round(args.limit * len(group) / len(rows))
            selected.extend(group[:count])
        selected = selected[: args.limit]
        rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"available": len(rows), "selected": len(selected),
                      "devices": {key: sum(row["device"] == key for row in selected)
                                  for key in sorted(groups)}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
