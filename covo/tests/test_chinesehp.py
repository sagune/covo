import unittest

from covo.chinesehp import make_sft_record


class ChineseHPSftTest(unittest.TestCase):
    def test_make_sft_record_edits(self):
        record = {
            "dataset": "chinesehp/aishell-1",
            "split": "dev",
            "id": "utt1",
            "reference": "广州市房地产中介协会分析",
            "nbest": [
                "广州市房地产中界协会分析",
                "广州市房地产中介协会分析",
            ],
            "nbest_pinyin": [
                "guang zhou shi fang di chan zhong jie xie hui fen xi",
                "guang zhou shi fang di chan zhong jie xie hui fen xi",
            ],
        }
        sft = make_sft_record(record, nbest_size=2)
        self.assertEqual(sft["id"], "utt1")
        self.assertEqual(len(sft["input"]["nbest"]), 2)
        self.assertIn("edits", sft["output"])
        self.assertGreaterEqual(len(sft["output"]["edits"]), 1)


if __name__ == "__main__":
    unittest.main()
