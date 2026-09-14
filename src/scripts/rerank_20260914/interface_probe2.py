#!/usr/bin/env python3
"""Old training template vs the template we render today."""
import json
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
R = Path("/root/autodl-tmp/.dsh_checks/rerank")


def brief(s, n=900):
    s = " ".join(str(s).split())
    return s[:n] + (" ..." if len(s) > n else "")


def dump(label, path):
    print("=" * 100)
    print(label)
    print(path)
    p = Path(path)
    if not p.exists():
        print("  MISSING")
        return
    with p.open(encoding="utf-8") as fh:
        r = json.loads(fh.readline())
    msgs = r.get("messages") or r.get("conversations") or []
    if msgs and isinstance(msgs[0], dict) and "from" in msgs[0]:
        msgs = [{"role": m.get("from"), "content": m.get("value")} for m in msgs]
    for m in msgs:
        c = m.get("content") or ""
        print("-" * 100)
        print("  [%s] %d chars" % (m.get("role"), len(c)))
        print("  " + brief(c, 1100).replace("\n", " | "))
    for k, v in r.items():
        if k in ("messages", "conversations"):
            continue
        print("  other key %-16s %s" % (k, brief(v, 200)))


print("### training_metadata.json, full ###")
for name in ("qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817",
             "qwen35_9b_aishell_hardneg_dropout_2epoch_20260815"):
    fp = TAR / "covo/outputs" / name / "training_metadata.json"
    print("-" * 100)
    print(name)
    print(json.dumps(json.loads(fp.read_text()), indent=2, ensure_ascii=False)[:2000])

dump("OLD ST-CMDS adapter training row (what the shipped adapter actually learned)",
     TAR / "covo/data/processed/stcmds_chinesehp_sensevoice/train.cb-hardnegative.qwen.jsonl")
dump("OLD AISHELL (base of the shipped adapter) training row",
     TAR / "covo/data/processed/chinesehp_aishell1/train_text_rewrite_hardneg_dropout.qwen.jsonl")
dump("NEW SFT row (what we are training on now)", R / "train_aishell_v1" / "train_sft.jsonl")
dump("OUR current inference prompt (V-A)", R / "e2eSTCMDS_VA.messages.jsonl")

print("=" * 100)
print("### training_args.bin max_length ###")
try:
    import torch
    for name in ("qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817",
                 "qwen35_9b_aishell_hardneg_dropout_2epoch_20260815"):
        fp = TAR / "covo/outputs" / name / "training_args.bin"
        a = torch.load(str(fp), map_location="cpu", weights_only=False)
        print("  %-56s max_length=%s epochs=%s lr=%s bs=%s accum=%s" % (
            name, getattr(a, "max_length", "?"), getattr(a, "num_train_epochs", "?"),
            getattr(a, "learning_rate", "?"), getattr(a, "per_device_train_batch_size", "?"),
            getattr(a, "gradient_accumulation_steps", "?")))
except Exception as e:
    print("  torch load failed: %r" % (e,))
