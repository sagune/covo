#!/usr/bin/env python3
"""CPU tests plus actual cached tokenizer/data checks; never loads 9B weights."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch

from connector import ConnectorConfig, TextAcousticResidual
from inputs import encode_prompt, prompt_and_span
from workflow import DEFAULT_LEGACY, HERE, SOURCE_FILES, legacy_module, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--report", type=Path, default=HERE / "CPU_PREFLIGHT.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior preflight report; choose a new report path")
    torch.set_num_threads(1)
    started = time.time()
    tests = unittest.defaultTestLoader.discover(str(HERE), pattern="test_connector.py")
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise RuntimeError("CPU unit tests failed")
    from transformers import AutoTokenizer
    p = legacy_module(args.legacy)
    tokenizer = AutoTokenizer.from_pretrained(p.BASE, trust_remote_code=True, local_files_only=True)
    ev = p.evaluator()
    report = dict(status="PASS_CPU_ONLY", tests_run=result.testsRun,
                  real_9b_gpu_smoke="NOT_RUN", training="NOT_STARTED", torch=torch.__version__,
                  sources={str(f): sha(f) for f in SOURCE_FILES + [HERE / "test_connector.py", HERE / "preflight.py"]})
    config = read(p.BASE / "config.json")
    hidden = config.get("hidden_size") or config["text_config"]["hidden_size"]
    net = TextAcousticResidual(ConnectorConfig(hidden))
    report["connector_parameters"] = sum(v.numel() for v in net.parameters())
    for split, n in (("train", 4096), ("dev", 1334)):
        rows = p.load(p.RUN / f"{split}.jsonl")
        if len(rows) != n:
            raise ValueError("Unexpected split size")
        max_prompt = 0
        for i, row in enumerate(rows):
            ids, selected = encode_prompt(p, tokenizer, row)
            target = json.dumps({"text": row["reference"]}, ensure_ascii=False) + tokenizer.eos_token
            if len(ids) + len(tokenizer(target, add_special_tokens=False)["input_ids"]) > 4096:
                raise ValueError(f"Overlong sequence: {row['id']}")
            if i == 0:
                prompt, _ = prompt_and_span(p, tokenizer, row)
                assert ids == tokenizer(prompt, add_special_tokens=False)["input_ids"]
                changed = dict(row, reference="PRIVATE_REFERENCE_NOT_INPUT")
                assert encode_prompt(p, tokenizer, changed) == (ids, selected)
            max_prompt = max(max_prompt, len(ids))
            if (i + 1) % 1000 == 0:
                print(json.dumps(dict(split=split, actual_tokenizer_checked=i + 1)), flush=True)
        report[split] = dict(rows=len(rows), max_prompt_tokens=max_prompt,
                             reference_chars=sum(len(ev.norm(r["reference"])) for r in rows))
    assert report["dev"]["reference_chars"] == 21102
    x, audio = torch.randn(12, hidden, dtype=torch.bfloat16), torch.randn(53, 512)
    assert torch.equal(net(x, audio), torch.zeros_like(x))
    report["hidden_size"] = hidden
    report["seconds"] = time.time() - started
    save(args.report, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
