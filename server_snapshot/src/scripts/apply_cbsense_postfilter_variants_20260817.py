#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.cbsensevoice_covo_bridge import post_filter_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    for suffix, same_length, anchor_digits in (
        ("same_length", True, False),
        ("anchor_digits", False, True),
        ("same_length_anchor_digits", True, True),
    ):
        output = Path(f"{args.output_prefix}_{suffix}.jsonl")
        shutil.copyfile(source, output)
        post_filter_predictions(
            argparse.Namespace(
                prediction_output=str(output),
                post_filter_same_length=same_length,
                post_filter_anchor_digits=anchor_digits,
            )
        )


if __name__ == "__main__":
    main()
