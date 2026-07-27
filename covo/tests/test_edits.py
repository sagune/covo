import unittest

from covo.edits import safe_apply_edits


class EditVerifierTest(unittest.TestCase):
    def test_safe_same_pinyin_edit(self):
        evidence = {
            "asr_top1": "广州市房地产中界协会分析",
            "nbest": ["广州市房地产中界协会分析", "广州市房地产中介协会分析"],
        }
        result = safe_apply_edits(
            evidence["asr_top1"],
            {"edits": [{"from": "中界", "to": "中介", "reason": "same_pinyin"}]},
            evidence,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["text"], "广州市房地产中介协会分析")

    def test_reject_unsupported_entity(self):
        evidence = {
            "asr_top1": "广州市房地产中界协会分析",
            "nbest": ["广州市房地产中界协会分析"],
        }
        result = safe_apply_edits(
            evidence["asr_top1"],
            {"edits": [{"from": "广州", "to": "上海"}]},
            evidence,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["text"], evidence["asr_top1"])

    def test_allow_deletion_edit(self):
        evidence = {
            "asr_top1": "而对面楼市成交抑制作用最大的限购",
            "nbest": ["而对面楼市成交抑制作用最大的限购"],
        }
        result = safe_apply_edits(
            evidence["asr_top1"],
            {"edits": [{"from": "面", "to": "", "reason": "delete_extra_char"}]},
            evidence,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["text"], "而对楼市成交抑制作用最大的限购")


if __name__ == "__main__":
    unittest.main()
