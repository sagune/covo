#!/usr/bin/env python3
"""Build the candidate-selector ablation data (target = a candidate INDEX).

Why: the main line trains the model to *write* the correct sentence, which trains
selection and editing together.  The selector variant instead emits `{"choice": k}`
over the numbered candidate list, which makes the task pure pool classification and
cannot hallucinate or break the JSON schema -- but it also forbids extra correction.
Running both gives the paper a clean decomposition of how much of the backend's
value comes from selecting versus from editing, and it is the documented fallback if
SFT on free text fails.

Target: the position, in the numbered list the model actually reads, of the
candidate with the smallest CER against the reference (earliest wins ties).  The
list is parsed straight out of the rendered prompt, so no assumption is made about
the ordering the bridge produced.

Rows where even the best visible candidate is far off are still kept: they teach
"pick the least bad", which is what a selector can do, and they are counted
separately so the ablation can be read honestly.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SYS = ("你是一个保守的中文 ASR 候选选择器。"
       "给定 ASR top-1、编号的 N-best 候选、拼音与 KWS 热词证据，"
       "从编号候选中选出最可信、最完整的一条。"
       "不要改写候选文本，不要拼接，不要新增候选。"
       "必须只输出一个合法 JSON 对象，格式为 {\"choice\": 编号}，编号从 1 开始。")

LINE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+\|\s")
# the user turn rendered for the text task ends by asking for {"text": ...}; a selector
# must be asked for the index instead, or the two instructions contradict each other
TAIL = re.compile(r"请输出\s*JSON\s*[:：]\s*\{[^}]*\}")
TAIL_NEW = '请输出 JSON：{"choice":编号}，编号从 1 开始。'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="rendered prompts with the candidates visible")
    ap.add_argument("--evidence", required=True, help="for the reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="")
    a = ap.parse_args()

    refs = {}
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            refs[(str(r.get("split")), str(r.get("id")))] = norm(r.get("reference") or "")

    rows, no_list, no_ref, far = [], 0, 0, 0
    best_cers = []
    for l in Path(a.messages).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        msgs = [m for m in (r.get("messages") or []) if m.get("role") != "assistant"]
        if len(msgs) < 2:
            continue
        user = msgs[1].get("content") or ""
        cands = []
        for line in user.splitlines():
            m = LINE.match(line)
            if m:
                cands.append((int(m.group(1)), m.group(2).strip()))
        if not cands:
            no_list += 1
            continue
        ref = norm(r.get("reference") or "") or refs.get((str(r.get("split")), str(r.get("id"))), "")
        if not ref:
            no_ref += 1
            continue
        cers = [(edit_distance(list(ref), list(norm(t))), pos) for pos, t in cands]
        best = min(cers)
        best_cers.append(best[0])
        if best[0] > 4:
            far += 1
        rows.append({"messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": TAIL.sub(TAIL_NEW, user)},
                                  {"role": "assistant",
                                   "content": json.dumps({"choice": best[1]}, ensure_ascii=False,
                                                         separators=(",", ":"))}],
                     "id": r.get("id"), "reference": ref,
                     "min_cer": best[0], "choice": best[1], "n_visible": len(cands)})

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    msg = ("rows %d | no numbered list %d | no reference %d | best visible CER > 4 edits %d (%.1f%%) | "
           "mean best visible CER %.3f | mean visible candidates %.1f" % (
               len(rows), no_list, no_ref, far, 100.0 * far / max(1, len(rows)),
               sum(best_cers) / max(1, len(best_cers)),
               sum(r["n_visible"] for r in rows) / max(1, len(rows))))
    print(msg)
    if a.stats:
        Path(a.stats).write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
