#!/usr/bin/env python3
"""Does the rendered INFERENCE prompt equal the training prompt minus the assistant turn?

iface_skeleton.py compared the system/user MESSAGES.  But what the model actually
conditions on is the chat-templated string, and the last piece - the generation prompt
tail, plus whether enable_thinking=False reproduces the empty `<think></think>` block the
SFT targets contain - was never checked.  If they differ, the model would be asked to
continue from a context it was never trained on, which is exactly the E41 failure in
miniature (and would silently cost the whole run).

Renders both sides with the model's own template and compares byte for byte.
"""
import json
import sys
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))
from transformers import AutoTokenizer   # noqa: E402

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
SFT = R / "train_aishell_v1" / "train_sft.jsonl"
ARMS = {
    "ST-CMDS V-A": R / "e2eSTCMDS_VA.messages.jsonl",
    "AISHELL dev": R / "train_aishell_v1" / "eval_aishell.messages.jsonl",
    "THCHS-30": R / "train_aishell_v1" / "eval_thchs.messages.jsonl",
}
tok = AutoTokenizer.from_pretrained(str(TAR / "models/Qwen3.5-9B"), trust_remote_code=True)


def render(msgs, gen):
    kw = {"tokenize": False, "add_generation_prompt": gen}
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


print("=" * 96)
print("training-side reference (first row of train_sft.jsonl)")
print("=" * 96)
with SFT.open(encoding="utf-8") as fh:
    tr = json.loads(fh.readline())
full_tr = render(tr["messages"], gen=False)
prompt_tr = render(tr["messages"][:2], gen=True)
print("  full (incl. assistant) tail: %r" % full_tr[-70:])
print("  prompt-only (gen=True) tail: %r" % prompt_tr[-70:])
print("  prompt-only is a PREFIX of full: %s" % full_tr.startswith(prompt_tr))
print("  the remainder (the target)    : %r" % full_tr[len(prompt_tr):][:70])

fail = 0
for name, path in ARMS.items():
    print()
    print("-" * 96)
    print(name)
    with path.open(encoding="utf-8") as fh:
        r = json.loads(fh.readline())
    p = render(r["messages"][:2], gen=True)
    same_tail = p.endswith(prompt_tr[-70:])
    print("  prompt tail: %r" % p[-70:])
    print("  tail identical to training's prompt tail: %s" % ("YES" if same_tail else "NO"))
    if not same_tail:
        fail += 1
    # and the training prompt for THIS row must be a prefix if the row has an assistant turn
    if len(r["messages"]) >= 3:
        f = render(r["messages"], gen=False)
        ok = f.startswith(p)
        print("  full render starts with the prompt: %s" % ("YES" if ok else "NO"))
        if not ok:
            fail += 1

print()
print("=" * 96)
print("VERDICT: %s" % ("inference conditions on exactly the training context"
                       if fail == 0 else "%d difference(s) - investigate" % fail))
print("=" * 96)
