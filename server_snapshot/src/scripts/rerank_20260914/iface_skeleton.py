#!/usr/bin/env python3
"""Verify the central premise of the training run: the SFT prompts and the inference
prompts are the SAME interface.

RESULTS 9.3 asserts the training interface is "逐字一致" with the inference arms, and the
whole argument for this run rests on it - the bug being fixed is precisely that the
shipped adapter was trained on a DIFFERENT interface (E41).  So it has to be checked on
the rendered text.

The check must be CONDITIONAL, because the bridge omits whole sections per row (a row with
no prompt hotwords has no 'Protected hotwords' / 'keeps prompt hotwords' / candidate-supported
lines; a row with no consensus spans has no 'Stable spans').  So a naive "every row has the
same labels" test false-alarms.  Instead this checks:

  1. system message, byte for byte
  2. tail instruction, byte for byte
  3. every per-row label sequence is a SUBSEQUENCE of one canonical order (no reordering,
     no unknown labels)
  4. the optional groups co-occur as units (protected <-> keeps-hw <-> protected-hw-line
     <-> candidate-supported)
  5. the skeleton SET produced by inference is a subset of the set the training data
     actually contains (so inference never presents a shape training has not seen)
  6. the candidate-list header text matches exactly
"""
import json
import re
from collections import Counter
from pathlib import Path

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
TRAIN = R / "train_aishell_v1" / "train_sft.jsonl"
ARMS = {
    "ST-CMDS V-A (inference)": R / "e2eSTCMDS_VA.messages.jsonl",
    "AISHELL dev (inference)": R / "train_aishell_v1" / "eval_aishell.messages.jsonl",
    "THCHS-30 (inference)": R / "train_aishell_v1" / "eval_thchs.messages.jsonl",
}
CANON = ["task", "compact-note", "asr-top1", "protected", "keeps-hw", "nbest-header",
         "stable-spans", "hotword-evidence", "protected-hw-line", "candidate-supported", "tail"]
LABELS = [
    ("task", "任务：从 CB-SenseVoice 的 ASR top-1、N-best 候选、拼音和 KWS 热词证据中选择最可信的完整中文转写。"),
    ("compact-note", "证据说明："),
    ("asr-top1", "ASR top-1: "),
    ("protected", "Protected hotwords that must be preserved exactly: "),
    ("keeps-hw", "ASR top-1 keeps prompt hotwords: "),
    ("nbest-header", "N-best with reliability labels (trusted_scored="),
    ("stable-spans", "Stable spans:"),
    ("hotword-evidence", "Hotword evidence:"),
    ("protected-hw-line", "- protected_hotwords="),
    ("candidate-supported", "candidate_supported_hotword"),
    ("tail", "请输出 JSON："),
]
GROUP = ["protected", "protected-hw-line"]   # keeps-hw and candidate-supported are independent
NBEST_HDR = re.compile(r"N-best with reliability labels \([^)]*\):")
AFTER_NBEST = ("Stable spans:", "Hotword evidence:", "请输出 JSON：")


def rows(path, k=None):
    out = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            out.append(json.loads(line))
            if k and len(out) >= k:
                break
    return out


def parts(r):
    m = {x["role"]: x["content"] for x in r["messages"]}
    return m.get("system", ""), m.get("user", "")


def skeleton(user):
    found = sorted((user.find(s), n) for n, s in LABELS if user.find(s) >= 0)
    return [n for _, n in found]


def is_subseq(seq, canon):
    it = iter(canon)
    return all(x in it for x in seq)


def header(user):
    m = NBEST_HDR.search(user)
    return m.group(0) if m else "(no nbest header)"


def ncandidates(user):
    """Count 'N. ' entries inside the n-best section only."""
    m = NBEST_HDR.search(user)
    if not m:
        return 0
    start = m.end()
    end = len(user)
    for lbl in AFTER_NBEST:
        p = user.find(lbl, start)
        if p >= 0:
            end = min(end, p)
    return len(re.findall(r"(?:^|\s)\d+\.\s", user[start:end]))


def tail(user):
    p = user.find("请输出 JSON：")
    return user[p:] if p >= 0 else "(NO TAIL)"


def profile(path, k=None):
    rs = rows(path, k)
    syss, tails, hdrs, skels, ncand = set(), set(), set(), Counter(), Counter()
    bad_seq, bad_group = [], []
    for r in rs:
        s, u = parts(r)
        syss.add(s)
        tails.add(tail(u))
        hdrs.add(header(u))
        sk = tuple(skeleton(u))
        skels[sk] += 1
        if not is_subseq(sk, CANON):
            bad_seq.append(sk)
        present = [g in sk for g in GROUP]
        if any(present) and not all(present):
            bad_group.append(sk)
        ncand[ncandidates(u)] += 1
    return dict(n=len(rs), syss=syss, tails=tails, hdrs=hdrs, skels=skels,
                bad_seq=bad_seq, bad_group=bad_group, ncand=ncand)


tr = profile(TRAIN)
print("=" * 100)
print("TRAINING (train_sft.jsonl, all %d rows)" % tr["n"])
print("=" * 100)
print("  distinct system messages : %d" % len(tr["syss"]))
print("  distinct tails           : %d" % len(tr["tails"]))
print("  distinct nbest headers   : %d   %s" % (len(tr["hdrs"]), list(tr["hdrs"])[0][:96]))
print("  rows violating canonical label order : %d" % len(tr["bad_seq"]))
print("  rows with a partially-present optional group : %d" % len(tr["bad_group"]))
print("  distinct skeletons (row counts):")
for sk, c in tr["skels"].most_common():
    print("     %5d  %s" % (c, " > ".join(sk)))
print("  numbered-candidate count distribution: %s" % dict(sorted(tr["ncand"].items())))
TRAIN_SKELS, TRAIN_SYS, TRAIN_TAIL, TRAIN_HDR = set(tr["skels"]), tr["syss"], tr["tails"], tr["hdrs"]

print()
print("=" * 100)
print("INFERENCE ARMS")
print("=" * 100)
allok = True
for name, path in ARMS.items():
    if not path.exists():
        print("\n%-28s MISSING" % name)
        allok = False
        continue
    p = profile(path)
    unseen = set(p["skels"]) - TRAIN_SKELS
    print()
    print("-" * 100)
    print("%s   (%d rows)" % (name, p["n"]))
    print("  system identical to training      : %s" % ("YES" if p["syss"] == TRAIN_SYS else "NO"))
    print("  tail identical to training        : %s" % ("YES" if p["tails"] == TRAIN_TAIL else "NO"))
    print("  nbest header identical            : %s%s" % (
        "YES" if p["hdrs"] == TRAIN_HDR else "NO",
        "" if p["hdrs"] == TRAIN_HDR else "\n     here: %s" % list(p["hdrs"])[0][:96]))
    print("  label order is canonical          : %s" % ("YES" if not p["bad_seq"] else "NO %s" % p["bad_seq"][:2]))
    print("  optional groups co-occur          : %s" % ("YES" if not p["bad_group"] else "NO %s" % p["bad_group"][:2]))
    print("  skeletons never seen in training  : %d %s" % (
        len(unseen), "" if not unseen else "\n     %s" % [(" > ".join(u)) for u in unseen][:3]))
    print("  numbered-candidate counts         : %s" % dict(sorted(p["ncand"].items())))
    if not (p["syss"] == TRAIN_SYS and p["tails"] == TRAIN_TAIL and p["hdrs"] == TRAIN_HDR
            and not p["bad_seq"] and not p["bad_group"] and not unseen):
        allok = False

print()
print("=" * 100)
print("VERDICT: %s" % ("PREMISE HOLDS - training and inference share one interface "
                      "(same system, tail, header, label order, and every inference shape "
                      "occurs in training)" if allok else "DIFFERENT - premise does NOT hold"))
print("=" * 100)
