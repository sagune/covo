#!/usr/bin/env python
"""Unit-test the prediction parser that feeds restore_eval.

`restore_eval.py` compares `prediction` against the reference character by
character, so this parser decides whether every downstream metric is meaningful.
It must return the sentence INSIDE the JSON wrapper, and must degrade gracefully
when the model emits prose, a fenced block, or nothing useful.
"""
import sys
from pathlib import Path

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "scripts"))
import infer_acoustic_covo as m  # noqa: E402

CASES = [
    ('{"text":"好英雄武器就是火麒麟不说了玩"}', "", "好英雄武器就是火麒麟不说了玩"),
    ('```json\n{"text":"好英雄武器"}\n```', "", "好英雄武器"),
    ('好的，结果是：{"text":"广州市房地产中介协会分析"} 希望有帮助', "", "广州市房地产中介协会分析"),
    ('{"text":""}', "回退文本", "回退文本"),
    ('完全是散文没有 JSON', "回退文本", "回退文本"),
    ('{"text":"多了一个大括号}}"}', "", "多了一个大括号}}"),
]

ok = 0
for raw, fb, want in CASES:
    got = m.parse_prediction(raw, fb)
    good = got == want
    ok += good
    print("[%s] %-46r -> %r" % ("PASS" if good else "FAIL", raw[:44], got))
    if not good:
        print("        expected %r" % want)
print()
print("%d/%d parse cases pass" % (ok, len(CASES)))
raise SystemExit(0 if ok == len(CASES) else 1)
