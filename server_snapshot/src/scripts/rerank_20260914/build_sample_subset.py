#!/usr/bin/env python3
"""Build the sampling probe subset.

Group A (damaged): rows where the greedy selector output destroyed a designated
hotword that the front-end had produced correctly. These are the rows where
"the LLM must hard-carry it"; we ask whether sampling ever recovers them.

Group B (control): rows with a designated hotword that greedy got right. We ask
whether sampling just adds noise and breaks them.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

WORK = Path("/root/autodl-tmp/.dsh_checks")
BASE = WORK / "aishell_selector"
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")

messages = [json.loads(x) for x in (BASE / "aishell_selector.messages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
preds = {json.loads(x)["id"]: json.loads(x) for x in (BASE / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

damaged, control = [], []
for r in messages:
    rid = r["id"]
    uid = pos2utt[int(rid)]
    kws = desig.get(uid, [])
    if not kws:
        continue
    inp = norm((r["input"] or {}).get("asr_top1") or "")
    pred = norm(preds[rid].get("prediction") or "")
    had = [k for k in kws if k in inp]
    if not had:
        continue
    lost = [k for k in had if k not in pred]
    if lost:
        damaged.append((r, uid, lost))
    else:
        control.append((r, uid, had))

rng = random.Random(20260914)
rng.shuffle(control)
control = control[:150]

subset = []
manifest = []
for tag, rows in (("damaged", damaged), ("control", control)):
    for r, uid, kws in rows:
        subset.append(r)
        manifest.append(dict(id=r["id"], uttid=uid, group=tag, hotwords=kws))

out = WORK / "sampling"
out.mkdir(parents=True, exist_ok=True)
(out / "subset.messages.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(damaged=len(damaged), control=len(control), total=len(subset)), ensure_ascii=False))
