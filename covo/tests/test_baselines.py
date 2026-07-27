import unittest

from covo.baselines import evaluate_baselines, oracle_nbest


class BaselinesTest(unittest.TestCase):
    def test_oracle_nbest_selects_best_candidate(self):
        record = {
            "reference": "房地产中介",
            "input": {
                "asr_top1": "房地产中界",
                "nbest": ["房地产中界", "房地产中介"],
            },
        }
        text, index, distance = oracle_nbest(record)
        self.assertEqual(text, "房地产中介")
        self.assertEqual(index, 1)
        self.assertEqual(distance, 0)

    def test_evaluate_baselines(self):
        metrics = evaluate_baselines(
            [
                {
                    "reference": "abc",
                    "input": {"asr_top1": "axc", "nbest": ["axc", "abc"]},
                }
            ]
        )
        self.assertEqual(metrics["samples"], 1)
        self.assertAlmostEqual(metrics["no_correction"]["cer"], 1 / 3)
        self.assertEqual(metrics["oracle_nbest"]["cer"], 0.0)


if __name__ == "__main__":
    unittest.main()
