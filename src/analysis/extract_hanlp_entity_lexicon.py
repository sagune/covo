#!/usr/bin/env python3
"""Extract person, location, and organization entities from JSONL references."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-keywords", type=int)
    parser.add_argument("--min-keywords", type=int, default=0)
    parser.add_argument("--include-jieba-terms", action="store_true")
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

    entity_keywords = sorted(
        (item for item in counts if counts[item] >= max(1, args.min_count)),
        key=lambda item: (-counts[item], item),
    )
    lexical_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    if args.include_jieba_terms:
        import jieba.posseg as pseg

        accepted_pos = {"n", "nr", "nrfg", "nrt", "ns", "nt", "nz", "vn", "eng"}
        chinese_term = re.compile(r"^[\u3400-\u9fff]{2,10}$")
        for text in tqdm(texts, desc="lexical terms"):
            for token in pseg.cut(text):
                word = str(token.word).strip()
                if token.flag in accepted_pos and chinese_term.fullmatch(word):
                    lexical_counts[word] += 1

        # Only used when segmentation cannot supply the requested vocabulary
        # size. Restricting this fallback to repeated 2-4 character spans keeps
        # the resulting contextual phrases compact and acoustically plausible.
        available = set(entity_keywords) | set(lexical_counts)
        if len(available) < max(0, args.min_keywords):
            for text in texts:
                chinese = "".join(re.findall(r"[\u3400-\u9fff]", text))
                for width in (2, 3, 4):
                    for start in range(max(0, len(chinese) - width + 1)):
                        fallback_counts[chinese[start:start + width]] += 1

    entity_set = set(entity_keywords)
    lexical_keywords = sorted(
        (item for item in lexical_counts if item not in entity_set),
        key=lambda item: (-lexical_counts[item], item),
    )
    known = entity_set | set(lexical_keywords)
    fallback_keywords = sorted(
        (item for item in fallback_counts if item not in known and fallback_counts[item] >= 2),
        key=lambda item: (-fallback_counts[item], item),
    )
    keywords = entity_keywords + lexical_keywords + fallback_keywords
    if args.max_keywords is not None:
        keywords = keywords[:max(0, args.max_keywords)]
    if len(keywords) < max(0, args.min_keywords):
        raise RuntimeError(
            f"only {len(keywords)} contextual terms were found; requested at least {args.min_keywords}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(keywords) + "\n", encoding="utf-8")
    summary = {
        "inputs": [str(path) for path in args.input],
        "utterances": len(texts),
        "entities": len(keywords),
        "min_count": max(1, args.min_count),
        "max_keywords": args.max_keywords,
        "min_keywords": max(0, args.min_keywords),
        "entity_terms": len(entity_keywords),
        "lexical_terms": len(lexical_keywords),
        "fallback_terms": len(fallback_keywords),
        "mentions": sum(counts.values()),
        "labels": dict(labels),
        "model": args.model,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
