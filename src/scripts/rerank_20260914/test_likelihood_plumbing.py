#!/usr/bin/env python3
"""Integration-test the likelihood-scoring arm WITHOUT the model forward pass.

Loading a 9B model is the only expensive part and it is also the least likely to be
wrong. Everything around it can be validated on CPU in seconds:

  1. the script's own helpers over all 5130 rows: prompt_messages drops the assistant turn
     and leaves 2 messages; unique_candidates returns [top1] + nbest deduped
  2. those candidates must equal the numbered list the prompt actually showed
  3. a synthetic predictions file in the script's exact output shape (prediction =
     candidate, selected_rank, candidate_scores) must be consumable by the whole
     downstream chain: restore_eval, gain_split, covo_error_anatomy, compare_arms
  4. the selected_rank reporting used by run_likelihood_arm.sh must work

If all four pass, the only untested piece of that arm is the forward pass itself.
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
MSG = R / "e2eSTCMDS_VA.messages.jsonl"
S = WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test"
PY = "/root/autodl-tmp/great/bin/python"
SCRIPT = WS / "src/analysis/covo_candidate_likelihood_rerank.py"
OUTSYN = R / "train_aishell_v1" / "lik_synthetic.jsonl"

spec = importlib.util.spec_from_file_location("lik", SCRIPT)
lik = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lik)

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm   # noqa: E402

LINE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+\|\s")

print("=" * 90)
print("1/2. helper behaviour and agreement with the rendered prompt")
print("=" * 90)
n = bad_pm = mismatch = fewer = 0
sizes = {}
rows = []
for line in MSG.open(encoding="utf-8"):
    if not line.strip():
        continue
    rec = json.loads(line)
    n += 1
    pm = lik.prompt_messages(rec)
    if len(pm) != 2 or pm[-1].get("role") != "user":
        bad_pm += 1
    cands = lik.unique_candidates(rec, 9)
    sizes[len(cands)] = sizes.get(len(cands), 0) + 1
    user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    shown = [norm(m.group(2)) for m in (LINE.match(x) for x in user.splitlines()) if m]
    if [norm(c) for c in cands] != shown:
        mismatch += 1
        if mismatch <= 2:
            print("   MISMATCH row %s: script=%s" % (rec.get("id"), [norm(c) for c in cands][:3]))
            print("                    prompt=%s" % shown[:3])
    if len(cands) < 9:
        fewer += 1
    if n <= 3:
        rows.append(rec)

print("  rows                                  : %d" % n)
print("  prompt_messages not (system,user)     : %d" % bad_pm)
print("  candidate set != the prompt's list    : %d   <-- must be 0" % mismatch)
print("  candidate-count distribution          : %s" % dict(sorted(sizes.items())))
print("  rows with fewer than 9 candidates     : %d" % fewer)
assert bad_pm == 0 and mismatch == 0, "plumbing broken"

print()
print("3. synthetic file in the script's output shape, through the whole downstream chain")
print("=" * 90)
OUTSYN.parent.mkdir(parents=True, exist_ok=True)
with OUTSYN.open("w", encoding="utf-8") as fh:
    for line in MSG.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        cands = lik.unique_candidates(rec, 9)
        fh.write(json.dumps({
            **rec,
            "prediction": cands[0],
            "selected_rank": 1,
            "candidate_scores": [{"rank": i + 1, "text": c, "lm_score": 0.0,
                                  "cb_score": 0.0, "combined_score": 0.0}
                                 for i, c in enumerate(cands)],
        }, ensure_ascii=False) + "\n")
print("  wrote %s (%d rows)" % (OUTSYN, sum(1 for _ in OUTSYN.open(encoding='utf-8'))))

for tool, extra in (
    ("restore_eval.py", ["--label", "SYNTHETIC rank-1 (= front end)",
                         "--aligned", str(S / "aligned.txt"), "--uttid-file", str(S / "uttid")]),
    ("gain_split.py", ["--label", "SYNTHETIC rank-1"]),
    ("covo_error_anatomy.py", ["--label", "SYNTHETIC rank-1"]),
):
    cmd = [PY, str(WS / ".dsh_checks" / tool), "--records", str(OUTSYN)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    key = [l for l in p.stdout.splitlines() if "restore[deployable]" in l or "input  CER" in l or "distance to" in l]
    print("  %-24s rc=%d  %s" % (tool, p.returncode, (key[0].strip() if key else p.stdout.strip()[:80])))
    if p.returncode != 0:
        print(p.stderr[-400:])

cmd = [PY, str(WS / ".dsh_checks" / "compare_arms.py"),
       "--a", str(R / "e2eSTCMDS_VC.predictions.jsonl"), "--b", str(OUTSYN),
       "--aligned", str(S / "aligned.txt"), "--uttid", str(S / "uttid"),
       "--label-a", "V-C", "--label-b", "SYNTHETIC", "--draws", "100"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("  %-24s rc=%d" % ("compare_arms.py", p.returncode))
print("     " + "\n     ".join(p.stdout.splitlines()[:4]))

print()
print("4. selected_rank reporting")
print("=" * 90)
import collections
c = collections.Counter()
for line in OUTSYN.open(encoding="utf-8"):
    c[json.loads(line).get("selected_rank")] += 1
print("  selected_rank: %s" % dict(c))
print()
print("ALL PLUMBING CHECKS PASSED - only the model forward pass remains untested")
