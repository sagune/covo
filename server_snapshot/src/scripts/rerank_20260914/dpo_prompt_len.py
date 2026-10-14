#!/usr/bin/env python3
"""Does the DPO fallback silently truncate its prompts?

train_lora_dpo_text.py's --max-prompt-length defaults to 896 and truncates FROM THE LEFT,
which removes the system message - the exact failure that silently degrades a run.
run_train_next.sh passes 2048, so the question is only whether the real prompts fit.

Measures every prompt in the DPO pair file with the model's own tokenizer and reports the
distribution, plus how many would be truncated at 896 / 1536 / 2048.
"""
import argparse
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))

from transformers import AutoTokenizer   # noqa: E402


def to_ids(tok, msgs):
    """Render to a string first, then tokenize.  apply_chat_template(tokenize=True) returns
    a BatchEncoding holding a per-message list on this tokenizer, which is easy to misread;
    rendering with tokenize=False and calling the tokenizer gives an unambiguous flat list."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/root/autodl-tmp/.dsh_checks/rerank/dpo_pairs_aishell.jsonl")
    ap.add_argument("--model", default=str(TAR / "models/Qwen3.5-9B"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    lens, types, n = [], {}, 0
    with open(a.pairs, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("prompt_messages") or []
            ids = to_ids(tok, msgs)
            if n == 0:
                # decisive validation: the decoded prompt must CONTAIN the user text
                dec = tok.decode(ids)
                probe = msgs[1]["content"][200:260] if len(msgs) > 1 else ""
                print("DIAGNOSTIC row 0: %d tokens; decoded %d chars; user content %d chars; "
                      "decoded contains a 60-char slice of the user message: %s"
                      % (len(ids), len(dec), len(msgs[1]["content"]), probe in dec))
                print("DIAGNOSTIC decoded head: %s" % dec[:150].replace("\n", " "))
            lens.append(len(ids))
            types[r.get("pair_type", "?")] = types.get(r.get("pair_type", "?"), 0) + 1
            n += 1
            if a.limit and n >= a.limit:
                break
    lens.sort()
    print("prompts measured: %d" % n)
    print("pair_type: %s" % types)
    print("prompt tokens: mean %.0f  p50 %d  p90 %d  p99 %d  max %d" % (
        sum(lens) / len(lens), lens[len(lens) // 2], lens[int(.90 * len(lens))],
        lens[int(.99 * len(lens))], lens[-1]))
    for cap in (896, 1024, 1536, 2048):
        over = sum(1 for x in lens if x > cap)
        print("  would be TRUNCATED at --max-prompt-length %4d : %5d prompts (%.1f%%)" % (
            cap, over, 100.0 * over / len(lens)))
    print()
    print("run_train_next.sh passes --max-prompt-length 2048 -> %s" % (
        "SAFE" if lens[-1] <= 2048 else "NOT SAFE: %d prompts exceed it" % sum(1 for x in lens if x > 2048)))


if __name__ == "__main__":
    main()
