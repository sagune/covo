#!/usr/bin/env python3
"""Render the SAME utterances in the interface the shipped adapters were trained on.

The shipped ST-CMDS adapter (qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817) learned:

  system 你是一个保守的中文ASR后纠错器。...证据不足时保持第一候选。(100 chars)
  user   N-best文本：\n1. ...\nN-best拼音：\n1. ...\n请输出：{"text":"纠错后的完整句子"}

while every arm in this session feeds it a 165-char "候选选择器 ... 不要默认保守复制 top-1"
system message and a completely restructured evidence block.  This script re-renders a
prompt file in the ORIGINAL template so the only thing that changes is the interface.

    render_old_iface.py --records <our .messages.jsonl> --style {stcmds,aishell} --out <file>

Candidate lists are taken verbatim from input.nbest / input.nbest_pinyin, so the control
differs from the new interface ONLY in the system message, the field framing and the tail.
"""
import argparse
import json
from pathlib import Path

SYS = {
    "stcmds": '你是一个保守的中文ASR后纠错器。根据N-best文本和拼音证据，输出纠错后的完整句子。'
              '只修正证据支持的识别错误；证据不足时保持第一候选。必须只输出JSON对象，格式为{"text":"完整句子"}。',
    "aishell": '你是一个保守的中文 ASR 后纠错器。根据 ASR top-1、N-best 候选、拼音和共识片段，'
               '输出最终纠错后的中文句子。必须只输出一个合法 JSON 对象，格式为 {"text":"..."}。'
               '不要输出解释、推理过程、Markdown 或额外字段。',
}
TAIL = '请输出：{"text":"纠错后的完整句子"}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="stcmds", choices=sorted(SYS))
    a = ap.parse_args()

    rows = []
    for line in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input") or {}
        nb = [t for t in (inp.get("nbest") or [])]
        py = [t for t in (inp.get("nbest_pinyin") or [])]
        txt = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(nb))
        pin = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(py)) if py else "(none)"
        user = "N-best文本：\n%s\nN-best拼音：\n%s\n%s" % (txt, pin, TAIL)
        rows.append({
            "id": r.get("id"),
            "dataset": r.get("dataset"),
            "split": r.get("split"),
            "reference": r.get("reference"),
            "input": inp,
            "messages": [
                {"role": "system", "content": SYS[a.style]},
                {"role": "user", "content": user},
                {"role": "assistant", "content": r.get("reference") or (nb[0] if nb else "")},
            ],
        })

    outp = Path(a.out)
    with outp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["messages"][1]["content"]) for r in rows]
    print("wrote %s rows=%d style=%s user_chars mean=%.0f max=%d" % (
        outp, len(rows), a.style, sum(lens) / max(1, len(lens)), max(lens) if lens else 0))


if __name__ == "__main__":
    main()
