import unittest

import torch

from model.sensevoice_ctc import ContextualHotwordScorer, ctc_prefix_beam_search


class SenseVoiceCTCTest(unittest.TestCase):
    def setUp(self):
        self.log_probs = torch.log(torch.tensor([
            [0.55, 0.40, 0.05],
            [0.55, 0.40, 0.05],
            [0.55, 0.05, 0.40],
        ], dtype=torch.float32))

    def test_neutral_prefix_beam(self):
        candidates = ctc_prefix_beam_search(self.log_probs, beam_size=4, token_topk=3)
        self.assertEqual(candidates[0].token_ids, (1,))
        self.assertEqual(candidates[1].token_ids, (1, 2))
        self.assertAlmostEqual(candidates[0].hotword_score, 0.0)

    def test_context_bias_promotes_complete_hotword(self):
        scorer = ContextualHotwordScorer([([1, 2], 1.0)], token_weight=1.0, completion_weight=1.0)
        candidates = ctc_prefix_beam_search(
            self.log_probs,
            beam_size=4,
            token_topk=3,
            hotword_scorer=scorer,
        )
        self.assertEqual(candidates[0].token_ids, (1, 2))
        self.assertGreater(candidates[0].hotword_score, 0.0)

    def test_repeated_label_requires_blank_transition(self):
        log_probs = torch.log(torch.tensor([
            [0.10, 0.90],
            [0.90, 0.10],
            [0.10, 0.90],
        ], dtype=torch.float32))
        candidates = ctc_prefix_beam_search(log_probs, beam_size=4, token_topk=2)
        self.assertEqual(candidates[0].token_ids, (1, 1))


if __name__ == "__main__":
    unittest.main()
