import unittest

from covo.formats import convert_record_format


class FormatTest(unittest.TestCase):
    def test_qwen_messages(self):
        record = {
            "id": "x",
            "instruction": "纠错",
            "input": {"asr_top1": "中界", "nbest": ["中界", "中介"]},
            "output": {"edits": [{"from": "中界", "to": "中介"}]},
        }
        converted = convert_record_format(record, "qwen-messages")
        self.assertEqual(converted["messages"][0]["role"], "system")
        self.assertEqual(converted["messages"][-1]["role"], "assistant")
        self.assertIn('"edits"', converted["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
