#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

BASE = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

raw = {json.loads(l)["id"]: json.loads(l) for l in (BASE / "dev.evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
prep = {json.loads(l)["id"]: json.loads(l) for l in (SEL / "aishell_selector.messages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

# locate the row whose normalised asr_top1 differs
target = None
for i in prep:
    a = raw[i]["input"].get("asr_top1") or ""
    b = prep[i]["input"].get("asr_top1") or ""
    if norm(a) != norm(b):
        target = i
        break

i = target
uid = pos2utt[int(i)]
ref = norm(raw[i].get("reference") or "")
a = norm(raw[i]["input"]["asr_top1"])
b = norm(prep[i]["input"]["asr_top1"])
print("id=%s  uttid=%s" % (i, uid))
print("reference (untouched): %r" % ref)
print("CB raw top-1         : %r   edits=%d" % (a, edit_distance(list(ref), list(a))))
print("after bridge         : %r   edits=%d" % (b, edit_distance(list(ref), list(b))))
print("designated hotwords for this utterance:", desig.get(uid))
print("hotword recalled in raw      :", [k for k in desig.get(uid, []) if k in a])
print("hotword recalled after bridge:", [k for k in desig.get(uid, []) if k in b])
print()
print("=> the bridge itself turns a correct output into a wrong one for this utterance.")

# quantify over the whole set: how many designated mentions are lost purely by the bridge?
lost_by_bridge = 0
gain_by_bridge = 0
for i in prep:
    uid = pos2utt[int(i)]
    a = norm(raw[i]["input"].get("asr_top1") or "")
    b = norm(prep[i]["input"].get("asr_top1") or "")
    for k in desig.get(uid, []):
        if k in a and k not in b:
            lost_by_bridge += 1
        if k not in a and k in b:
            gain_by_bridge += 1
print("designated mentions destroyed by the bridge rewrite: %d" % lost_by_bridge)
print("designated mentions created  by the bridge rewrite: %d" % gain_by_bridge)
