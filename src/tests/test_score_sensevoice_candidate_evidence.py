import unittest

import torch

from analysis.score_sensevoice_candidate_evidence import ctc_sequence_score


class SenseVoiceCandidateEvidenceTest(unittest.TestCase):
    def test_ctc_score_prefers_frame_supported_token(self):
        probabilities = torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        log_probs = probabilities.log()
        self.assertGreater(
            ctc_sequence_score(log_probs, [1], blank_id=0),
            ctc_sequence_score(log_probs, [2], blank_id=0),
        )

    def test_empty_candidate_is_rejected(self):
        log_probs = torch.tensor([[0.8, 0.2]], dtype=torch.float32).log()
        self.assertEqual(ctc_sequence_score(log_probs, [], blank_id=0), -1e9)


if __name__ == "__main__":
    unittest.main()
