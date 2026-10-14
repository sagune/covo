import tempfile
import unittest
from pathlib import Path

from covo.io import read_jsonl, write_jsonl


class IoTest(unittest.TestCase):
    def test_roundtrip_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            count = write_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(count, 2)
            self.assertEqual(list(read_jsonl(path)), [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()
