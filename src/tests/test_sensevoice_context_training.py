import unittest

import torch

from analysis.train_sensevoice_context_adapter import phrase_ctc_ranking_loss


class PhraseCtcRankingLossTest(unittest.TestCase):
    def test_prefers_correct_complete_phrase(self):
        blank = 0
        correct_logits = torch.tensor(
            [
                [[-4.0, 4.0, -4.0, -4.0], [4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, -4.0]]
            ],
            requires_grad=True,
        )
        wrong_logits = torch.tensor(
            [
                [[-4.0, 4.0, -4.0, -4.0], [4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, 4.0]]
            ],
            requires_grad=True,
        )
        correct_loss = phrase_ctc_ranking_loss(
            correct_logits.log_softmax(-1),
            [1, 2],
            [[1, 3]],
            blank,
            margin=0.3,
        )
        wrong_loss = phrase_ctc_ranking_loss(
            wrong_logits.log_softmax(-1),
            [1, 2],
            [[1, 3]],
            blank,
            margin=0.3,
        )
        self.assertLess(float(correct_loss.detach()), float(wrong_loss.detach()))
        wrong_loss.backward()
        self.assertIsNotNone(wrong_logits.grad)


if __name__ == "__main__":
    unittest.main()
