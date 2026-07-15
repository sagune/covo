import sys
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from cbwhisper_covo_bridge import simplify_input_block  # noqa: E402


class CovoBridgeMetadataTest(unittest.TestCase):
    def test_simplify_preserves_keyword_mentions(self):
        block = simplify_input_block({
            "keyword_mentions": [{
                "mention": "許瑋寧",
                "total_offset": 0,
                "end_offset": 3,
                "ner_tag": "PER",
            }],
        })

        self.assertEqual(block["keyword_mentions"], [{
            "mention": "许玮宁",
            "total_offset": 0,
            "end_offset": 3,
            "ner_tag": "PER",
        }])


if __name__ == "__main__":
    unittest.main()
