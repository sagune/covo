import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from cbwhisper_covo_bridge import parse_args, prepare_records, simplify_input_block  # noqa: E402


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

    def test_prepare_accepts_generic_prediction_as_top1(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sensevoice.jsonl"
            output_path = Path(directory) / "messages.jsonl"
            input_path.write_text(
                json.dumps({
                    "id": "utt-1",
                    "reference": "水利工程",
                    "prediction": "水利工程。",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            argv = [
                "cbwhisper_covo_bridge.py",
                "prepare",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv):
                record = next(iter(prepare_records(parse_args())))

        self.assertEqual(record["input"]["asr_top1"], "水利工程。")
        self.assertEqual(record["input"]["nbest"], ["水利工程。"])


if __name__ == "__main__":
    unittest.main()
