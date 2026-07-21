import unittest

import torch

from model.sensevoice_context_adapter import SenseVoiceContextAdapter


class SenseVoiceContextAdapterTest(unittest.TestCase):
    def test_empty_context_is_identity(self):
        adapter = SenseVoiceContextAdapter(hidden_size=4, projection_size=4)
        hidden = torch.randn(1, 3, 4)
        logits = torch.log_softmax(torch.randn(1, 3, 6), dim=-1)
        output = adapter(hidden, logits, torch.randn(6, 4), [])
        self.assertTrue(torch.equal(output, logits))

    def test_matching_context_token_is_boosted(self):
        adapter = SenseVoiceContextAdapter(hidden_size=2, projection_size=2)
        with torch.no_grad():
            adapter.audio_projection.weight.copy_(torch.eye(2))
            adapter.token_projection.weight.copy_(torch.eye(2))
            adapter.log_scale.fill_(2.0)
            adapter.log_temperature.fill_(2.0)
            adapter.similarity_offset.zero_()
        hidden = torch.tensor([[[1.0, 0.0]]])
        logits = torch.log_softmax(torch.zeros(1, 1, 3), dim=-1)
        weights = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        output = adapter(hidden, logits, weights, [1, 2], [1.0, 1.0])
        self.assertGreater(float(output[0, 0, 1]), float(output[0, 0, 2]))

    def test_adapter_receives_gradients(self):
        adapter = SenseVoiceContextAdapter(hidden_size=4, projection_size=2)
        output = adapter(
            torch.randn(1, 3, 4),
            torch.log_softmax(torch.randn(1, 3, 5), dim=-1),
            torch.randn(5, 4),
            [1, 2],
        )
        (-output[..., 1].mean()).backward()
        self.assertIsNotNone(adapter.audio_projection.weight.grad)

    def test_position_logits_share_context_alignment(self):
        adapter = SenseVoiceContextAdapter(hidden_size=2, projection_size=2)
        with torch.no_grad():
            adapter.audio_projection.weight.copy_(torch.eye(2))
            adapter.token_projection.weight.copy_(torch.eye(2))
            adapter.log_temperature.fill_(2.0)
            adapter.similarity_offset.zero_()
        hidden = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        logits = torch.log_softmax(torch.zeros(1, 2, 3), dim=-1)
        weights = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        _, position_logits = adapter(hidden, logits, weights, [1], return_position_logits=True)
        self.assertGreater(float(position_logits[0, 0]), float(position_logits[0, 1]))


if __name__ == "__main__":
    unittest.main()
