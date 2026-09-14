#!/usr/bin/env python3
"""Is the shipped ST-CMDS adapter simply mis-calibrated to the old prompt format?

Reads the first row of the Qwen-format training files the shipped adapters were trained
on and compares their message template against the prompts we render today.
CPU only - safe to run while training owns the GPU.
"""
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
COVO = TAR / "covo"
R = Path("/root/autodl-tmp/.dsh_checks/rerank")


def brief(s, n=180):
    s = " ".join(str(s).split())
    return s[:n] + (" ..." if len(s) > n else "")


def show_msgs(label, msgs):
    print("  %-34s n_msgs=%d" % (label, len(msgs)))
    for m in msgs:
        c = m.get("content") or ""
        print("      [%-9s] %5d chars | %s" % (m.get("role"), len(c), brief(c, 110)))


def probe_file(label, path, key_msgs=None, limit=1):
    p = Path(path)
    if not p.exists():
        print("  %-34s MISSING (%s)" % (label, p))
        return
    print("  %-34s %s" % (label, p))
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get(key_msgs) if key_msgs else None
            if msgs is None:
                msgs = r.get("messages") or r.get("conversations") or []
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "from" in msgs[0]:
                msgs = [{"role": m.get("from"), "content": m.get("value")} for m in msgs]
            show_msgs(label, msgs)
            for k in ("prompt", "response", "target", "text", "input", "output"):
                if k in r and isinstance(r[k], str):
                    print("      key %-9s %5d chars | %s" % (k, len(r[k]), brief(r[k], 110)))


print("=" * 100)
print("1) shipped adapters: their own training data (what prompt did they learn?)")
print("=" * 100)
cands = sorted(COVO.glob("outputs/*/training_metadata.json"))
for c in cands:
    try:
        md = json.loads(c.read_text())
    except Exception as e:
        md = {"error": str(e)}
    print("  %-70s %s" % (c.parent.name, brief(json.dumps(md, ensure_ascii=False), 200)))
print()
for name in ("qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817",
             "qwen35_9b_aishell_hardneg_dropout_2epoch_20260815"):
    d = COVO / "outputs" / name
    print("-" * 100)
    print("adapter dir: %s  exists=%s" % (d, d.exists()))
    for f in sorted(d.glob("training_args*")):
        print("    %s (%d bytes)" % (f.name, f.stat().st_size))
    for f in ("training_metadata.json", "train_config.json", "adapter_config.json"):
        fp = d / f
        if fp.exists():
            print("    %s: %s" % (f, brief(fp.read_text(), 300)))
print()

print("=" * 100)
print("2) the raw instruction files those adapters were trained from")
print("=" * 100)
for pat in ("stcmds_chinesehp_sensevoice/*.jsonl", "aishell*/*hardneg*.jsonl"):
    for f in sorted(Path("/root/autodl-tmp").glob("**/" + pat))[:6]:
        n = sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
        print("  %-88s %7d rows" % (f, n))
print()

print("=" * 100)
print("3) the prompts we render today")
print("=" * 100)
for lab, f in (("our ST-CMDS V-A prompt", R / "e2eSTCMDS_VA.messages.jsonl"),
               ("our AISHELL-dev prompt", R / "train_aishell_v1" / "eval_aishell.messages.jsonl"),
               ("our THCHS-30 prompt", R / "train_aishell_v1" / "eval_thchs.messages.jsonl"),
               ("our NEW SFT training row", R / "train_aishell_v1" / "train_sft.jsonl")):
    print("-" * 100)
    probe_file(lab, f, key_msgs=None, limit=1)
