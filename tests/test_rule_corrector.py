import unittest

from covo.rule_corrector import apply_rule_correction, make_rule_edits


class RuleCorrectorTest(unittest.TestCase):
    def test_rule_uses_close_nbest_candidate(self):
        evidence = {
            "asr_top1": "广州市房地产中界协会分析",
            "nbest": ["广州市房地产中界协会分析", "广州市房地产中介协会分析", "广州市房地产中介协会分析"],
        }
        edits = make_rule_edits(evidence, max_char_distance=2, max_pinyin_distance=1)
        self.assertEqual(edits[0].from_text, "界")
        self.assertEqual(edits[0].to_text, "介")
        result = apply_rule_correction(evidence, max_char_distance=2, max_pinyin_distance=1)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["text"], "广州市房地产中介协会分析")


if __name__ == "__main__":
    unittest.main()
