#!/usr/bin/env python
"""Unpack one file from a ZIP archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.archives import unpack_zip_member


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ZIP archive path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--member", default="", help="ZIP member name. Defaults to the only member.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = unpack_zip_member(args.input, args.output, member=args.member, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
