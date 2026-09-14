#!/usr/bin/env bash
# Finish the CPU-only bridge dry run for ST-CMDS at the original admission, and
# render one row where the reranked winner originally sat outside the top-8 so the
# interface mismatch of section 7.9 can be read directly off the real prompt.
set -uo pipefail
WS=/root/autodl-tmp
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
LOG="$R/bridge_dryrun.log"

[[ -s "$R/stcmds_orig_fixed.jsonl" ]] || cp -f "$R/tmp_st.jsonl" "$R/stcmds_orig_fixed.jsonl"
echo "stcmds_orig_fixed rows=$(wc -l < "$R/stcmds_orig_fixed.jsonl")" | tee -a "$LOG"

cd "$WS"
COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)
MSG="$R/e2eSTCMDSORIGFIX.messages.jsonl"
if [[ ! -s "$MSG" ]]; then
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_orig_fixed.jsonl" \
    --output "$MSG.inprogress" "${COMMON[@]}" >> "$LOG" 2>&1
  [[ -s "$MSG.inprogress" ]] && mv -f "$MSG.inprogress" "$MSG"
fi
echo "OK    e2eSTCMDSORIGFIX.messages.jsonl rows=$(wc -l < "$MSG" 2>/dev/null || echo 0) (expected 5130)" | tee -a "$LOG"

"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
src = {}
for l in (R / "tmp_st.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        src[str(r["id"])] = r

def prompt_text(rec):
    msgs = rec.get("messages") or []
    c = msgs[1].get("content") if len(msgs) > 1 else (msgs[0].get("content") if msgs else "")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""

shown = 0
for l in (R / "e2eSTCMDSORIGFIX.messages.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    rec = json.loads(l)
    rid = str(rec.get("id"))
    s = src.get(rid)
    if s is None:
        continue
    inp = s["input"]
    top = norm(inp.get("asr_top1") or "")
    ranks = {}
    for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
        if c.get("text"):
            ranks.setdefault(norm(c["text"]), int(c.get("rank", 999)))
    wr = ranks.get(top)
    if wr is None or wr <= 8:
        continue
    txt = prompt_text(rec)
    head = txt.split("N-best with reliability labels")[0]
    nb = txt.split("N-best with reliability labels", 1)[1] if "N-best with reliability labels" in txt else ""
    print("=" * 78)
    print("row id=%s | reranked winner originally rank=%d" % (rid, wr))
    print("--- ASR top-1 line as COVO sees it ---")
    for line in head.splitlines():
        if line.startswith("ASR top-1"):
            print("   ", line)
    print("--- first 3 N-best lines ---")
    for line in [x for x in nb.splitlines() if x.strip()][:3]:
        print("   ", line[:170])
    print("--- candidate-scores block (original rank order) ---")
    if "CB-SenseVoice candidate scores" in txt:
        blk = txt.split("CB-SenseVoice candidate scores", 1)[1]
        for line in [x for x in blk.splitlines() if x.strip()][:4]:
            print("   ", line[:170])
    shown += 1
    if shown >= 2:
        break
print("=" * 78)
print("rows shown:", shown)
PY
touch "$R/BRIDGE_DRYRUN_STCMDS_DONE"
