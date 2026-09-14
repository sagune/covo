#!/usr/bin/env python3
"""Turn the rendered AISHELL-train prompts into a Qwen-messages SFT file.

Input : train_aishell.messages.jsonl   (system + user, rendered by the same bridge
                                        and flags the inference arms use)
        train_aishell_evidence.jsonl   (for the reference)
Output: train_aishell_sft.jsonl        (system + user + assistant)

Target = the normalised reference, serialised exactly as inference expects:
    {"text":"..."}   with separators (",", ":") and ensure_ascii=False
which matches the assistant strings in the existing COVO SFT data.
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref_by_key = {}
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ref_by_key[(str(r.get("split")), str(r.get("id")))] = norm(r.get("reference") or "")

    rows, skipped, empty = [], 0, 0
    for l in Path(a.messages).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        ref = norm(r.get("reference") or "")
        if not ref:
            ref = ref_by_key.get((str(r.get("split")), str(r.get("id"))), "")
        if not ref:
            skipped += 1
            continue
        msgs = [m for m in (r.get("messages") or []) if m.get("role") != "assistant"]
        if len(msgs) < 2:
            empty += 1
            continue
        msgs.append({"role": "assistant",
                     "content": json.dumps({"text": ref}, ensure_ascii=False, separators=(",", ":"))})
        rows.append({"messages": msgs, "id": r.get("id"), "reference": ref})

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    print("wrote %d rows to %s (skipped no-ref %d, malformed %d)" % (len(rows), a.out, skipped, empty))
    if rows:
        s = rows[0]["messages"]
        print("  system len %d | user len %d | assistant %s" % (
            len(s[0]["content"]), len(s[1]["content"]), s[2]["content"][:80]))


if __name__ == "__main__":
    main()
