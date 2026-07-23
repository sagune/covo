import unittest

import torch

from model.sensevoice_context_adapter import (
    SenseVoiceContextAdapter,
    SenseVoicePhraseContextAdapter,
    pinyin_bucket_ids,
)


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


class SenseVoicePhraseContextAdapterTest(unittest.TestCase):
    def test_pinyin_hash_is_stable_and_aligned(self):
        first = pinyin_bucket_ids("邓郁松", 3)
        second = pinyin_bucket_ids("邓郁松", 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(0 < item < 512 for item in first))

    def test_phrase_adapter_shapes_and_gradients(self):
        adapter = SenseVoicePhraseContextAdapter(
            hidden_size=8,
            projection_size=8,
            num_heads=2,
            num_context_layers=1,
            max_phrase_tokens=4,
            dropout=0.0,
        )
        hidden = torch.randn(1, 5, 8)
        weights = torch.randn(12, 8)
        bias = torch.randn(12)
        base = torch.log_softmax(torch.randn(1, 5, 12), dim=-1)
        output, positions, phrases = adapter(
            hidden,
            base,
            weights,
            context_phrases=[[2, 3], [4, 5, 6]],
            context_pinyin_ids=[[11, 12], [13, 14, 15]],
            ctc_token_bias=bias,
            return_auxiliary=True,
        )
        self.assertEqual(tuple(output.shape), (1, 5, 12))
        self.assertEqual(tuple(positions.shape), (1, 5, 2))
        self.assertEqual(tuple(phrases.shape), (1, 2))
        (-output[..., 2].mean() + positions.mean() + phrases.mean()).backward()
        self.assertIsNotNone(adapter.cross_attention.in_proj_weight.grad)

    def test_monotonic_phrase_activation_shapes_and_gradients(self):
        adapter = SenseVoicePhraseContextAdapter(
            hidden_size=8,
            projection_size=8,
            num_heads=2,
            num_context_layers=1,
            dropout=0.0,
            monotonic_phrase_activation=True,
        )
        hidden = torch.randn(1, 12, 8, requires_grad=True)
        output, positions, phrases = adapter(
            hidden,
            torch.randn(1, 12, 16).log_softmax(-1),
            torch.randn(16, 8),
            context_phrases=[[2, 3, 4], [7, 8]],
            context_pinyin_ids=[[11, 12, 13], [21, 22]],
            return_auxiliary=True,
        )
        self.assertEqual(tuple(output.shape), (1, 12, 16))
        self.assertEqual(tuple(positions.shape), (1, 12, 2))
        self.assertEqual(tuple(phrases.shape), (1, 2))
        output.sum().backward()
        self.assertIsNotNone(hidden.grad)

    def test_monotonic_residual_floor_is_serialized(self):
        adapter = SenseVoicePhraseContextAdapter(
            hidden_size=8,
            projection_size=8,
            num_heads=2,
            num_context_layers=1,
            monotonic_phrase_activation=True,
            monotonic_residual_floor=0.75,
        )
        self.assertEqual(adapter.config()["monotonic_residual_floor"], 0.75)


if __name__ == "__main__":
    unittest.main()
