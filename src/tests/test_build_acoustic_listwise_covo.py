import unittest

from analysis.build_acoustic_listwise_covo import build_record


class BuildAcousticListwiseCovoTest(unittest.TestCase):
    def test_scores_are_aligned_and_length_normalized(self):
        source = {
            "id": "utt-1",
            "reference": "饮水工程",
            "input": {
                "asr_top1": "引水工程",
                "nbest": ["引水工程", "饮水工程"],
                "nbest_pinyin": ["yin shui gong cheng", "yin shui gong cheng"],
                "hotwords": [{"text": "饮水工程", "score": 0.9}],
                "cbwhisper": {
                    "candidates": [
                        {
                            "text": "引水工程",
                            "token_ids": [1, 2, 3, 4],
                            "asr_score": -4.0,
                            "search_score": -3.0,
                            "source": "sensevoice_neutral",
                        },
                        {
                            "text": "饮水工程",
                            "token_ids": [5, 2, 3, 4],
                            "asr_score": -2.0,
                            "search_score": -1.0,
                            "ctc_hotword_score": 1.0,
                            "source": "sensevoice_hotword",
                        },
                    ]
                },
            },
        }
        record = build_record(source, max_nbest=10, max_hotwords=8, require_reference_candidate=True)
        self.assertIsNotNone(record)
        self.assertEqual(record["target_index"], 1)
        self.assertEqual(record["candidate_evidence"][1]["acoustic_rank"], 1)
        self.assertAlmostEqual(record["candidate_evidence"][1]["acoustic_score_norm"], -0.5)
        self.assertIn("候选分歧片段", record["messages"][1]["content"])
        self.assertIn("声学排名=1/2", record["messages"][1]["content"])

    def test_reference_outside_nbest_can_be_rejected(self):
        source = {
            "reference": "饮水工程",
            "input": {"nbest": ["引水工程"], "nbest_pinyin": ["yin shui gong cheng"]},
        }
        self.assertIsNone(build_record(source, 10, 8, require_reference_candidate=True))


if __name__ == "__main__":
    unittest.main()
