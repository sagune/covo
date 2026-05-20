import unittest

from covo.model_output import parse_model_edits_json


class ModelOutputParserTest(unittest.TestCase):
    def test_parse_fenced_json(self):
        text = '这里是结果：```json\n{"edits":[{"from":"中界","to":"中介"}]}\n```'
        parsed = parse_model_edits_json(text)
        self.assertEqual(parsed["edits"][0]["from"], "中界")
        self.assertEqual(parsed["parse_warnings"], [])

    def test_parse_invalid_as_empty(self):
        parsed = parse_model_edits_json("不需要修改")
        self.assertEqual(parsed["edits"], [])
        self.assertIn("no_valid_json_edits", parsed["parse_warnings"])


if __name__ == "__main__":
    unittest.main()
