import unittest

from analysis.covo_candidate_likelihood_rerank import cb_scores, unique_candidates


class CovoCandidateLikelihoodRerankTest(unittest.TestCase):
    def test_candidates_are_normalized_and_deduplicated(self):
        record = {
            "input": {
                "asr_top1": "引水工程。",
                "nbest": ["引水工程", "饮水工程", "引水工程。"],
            }
        }
        self.assertEqual(unique_candidates(record, 6), ["引水工程。", "饮水工程"])

    def test_cb_scores_follow_candidate_text(self):
        record = {
            "input": {
                "cbwhisper": {
                    "candidates": [
                        {"text": "引水工程", "total_score": 0.8},
                        {"text": "饮水工程", "total_score": 0.3},
                    ]
                }
            }
        }
        self.assertEqual(cb_scores(record, ["引水工程。", "饮水工程"]), [0.8, 0.3])


if __name__ == "__main__":
    unittest.main()
