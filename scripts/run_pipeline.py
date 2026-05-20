#!/usr/bin/env python
"""Run a JSON/YAML pipeline config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.config import load_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", help="Override config log_file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results = run_pipeline(config, dry_run=args.dry_run, log_file=args.log_file)
    print(json.dumps({"config": args.config, "dry_run": args.dry_run, "steps": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
