import tempfile
import unittest
from pathlib import Path

from analysis.build_shuili_video_dataset import load_explicit_hotwords


class ShuiliHotwordExpansionTest(unittest.TestCase):
    def test_explicit_hotwords_are_deduplicated_and_corpus_backed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hotwords.txt"
            path.write_text(
                "# diagnostic terms\n年径流\n年径流\n不存在术语\nTBM\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_explicit_hotwords(str(path), ["黄河年径流量发生变化"]),
                ["年径流"],
            )


if __name__ == "__main__":
    unittest.main()
