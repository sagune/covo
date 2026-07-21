#!/usr/bin/env python3
"""Extract person, location, and organization entities from JSONL references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def iter_batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--model", default="CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH")
    args = parser.parse_args()

    import hanlp

    model_url = getattr(
        hanlp.pretrained.mtl,
        args.model,
        getattr(hanlp.pretrained.ner, args.model, args.model),
    )
    recognizer = hanlp.load(model_url)
    texts = []
    for path in args.input:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(str(json.loads(line).get("reference", "")).strip())

    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    accepted_labels = {"NR", "NS", "NT", "PERSON", "LOCATION", "ORGANIZATION"}
    for batch in tqdm(iter_batches(texts, max(1, args.batch_size)), total=(len(texts) + args.batch_size - 1) // args.batch_size):
        outputs = recognizer(batch, tasks="ner")
        if isinstance(outputs, dict):
            outputs = outputs.get("ner/msra", next((value for key, value in outputs.items() if key.startswith("ner/")), []))
        elif len(batch) == 1 and outputs and isinstance(outputs[0], tuple):
            outputs = [outputs]
        for entities in outputs:
            for entity in entities:
                mention = str(entity[0]).strip()
                label = str(entity[1]).upper()
                if label in accepted_labels and len(mention) >= args.min_chars:
                    counts[mention] += 1
                    labels[label] += 1

    keywords = sorted(counts, key=lambda item: (-counts[item], item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(keywords) + "\n", encoding="utf-8")
    summary = {
        "inputs": [str(path) for path in args.input],
        "utterances": len(texts),
        "entities": len(keywords),
        "mentions": sum(counts.values()),
        "labels": dict(labels),
        "model": args.model,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
