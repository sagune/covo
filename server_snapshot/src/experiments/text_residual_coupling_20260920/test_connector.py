"""CPU engineering tests; no model downloads, training jobs or dev selection."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from connector import ConnectorConfig, TextAcousticResidual, pool_frames
from inputs import encode_prompt, prompt_and_span, residual_batch
from diagnostics import diagnose, edit_counts
from workflow import audio_path, verify

torch.set_num_threads(1)


class FakeTokenizer:
    is_fast = True
    eos_token = "~"

    def apply_chat_template(self, messages, **kwargs):
        return "[system]" + messages[0]["content"] + "[user]" + messages[1]["content"] + "[assistant]"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": [ord(c) % 512 for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class FakeLegacy:
    @staticmethod
    def evidence(row):
        inp = row["input"]
        return {"asr_top1": inp["asr_top1"], "nbest": inp.get("nbest", [])[:8],
                "hotwords": inp.get("prompt_hotwords", [])[:32]}

    @classmethod
    def messages(cls, row, stage, acoustic):
        assert stage == "B" and not acoustic
        return [{"role": "system", "content": "仅输出JSON"},
                {"role": "user", "content": "请校正。识别证据：" + json.dumps(cls.evidence(row), ensure_ascii=False)}]


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, 16)
        self.head = nn.Linear(16, 512)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None):
        # A frozen causal toy decoder, enough to test the connector-to-loss graph.
        count = torch.arange(1, inputs_embeds.shape[1] + 1, device=inputs_embeds.device)[None, :, None]
        logits = self.head(inputs_embeds.cumsum(1) / count)
        loss = None if labels is None else nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512),
                                                                       labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def row(top1='记者乔加伟抵达北京'):
    return {"id": "fixture_001", "reference": "PRIVATE_REFERENCE", "input": {
        "asr_top1": top1, "nbest": [top1, "记者乔佳伟到达北京"], "prompt_hotwords": ["乔加伟"],
        "oracle_hotwords": ["PRIVATE_ORACLE"], "keyword_mentions": ["PRIVATE_MENTION"]}}


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(917)
        self.net = TextAcousticResidual(ConnectorConfig(16, audio_size=8, width=16, heads=4))
        self.model = TinyLM()
        self.tokenizer = FakeTokenizer()
        self.frames = torch.randn(13, 8)

    def batch(self, example=None, mode="true", training=False):
        return residual_batch(FakeLegacy, self.model, self.tokenizer, self.net, example or row(),
                              self.frames, mode, training)

    def test_pool_remainder_and_short_utterance(self):
        x = torch.arange(40.).reshape(5, 8)
        self.assertTrue(torch.equal(pool_frames(x), torch.stack([x[:4].mean(0), x[4]])))
        self.assertTrue(torch.equal(pool_frames(x[:2]), x[:2].mean(0, keepdim=True)))

    def test_bad_audio_is_not_silently_dropped(self):
        for x in (torch.empty(0, 8), torch.full((2, 8), float("nan"))):
            with self.assertRaises(ValueError):
                pool_frames(x)

    def test_zero_init_exact_text_embeddings_and_logits(self):
        true, off = self.batch(), self.batch(mode="off")
        self.assertTrue(torch.equal(true["inputs_embeds"], off["inputs_embeds"]))
        self.assertTrue(torch.equal(self.model(**true).logits, self.model(**off).logits))

    def test_off_does_not_require_audio_or_connector(self):
        result = residual_batch(FakeLegacy, self.model, self.tokenizer, None, row(), None, "off")
        self.assertTrue(torch.equal(result["inputs_embeds"], self.batch(mode="text")["inputs_embeds"]))

    def test_off_still_equals_text_after_training_changes(self):
        nn.init.normal_(self.net.output.weight)
        self.assertTrue(torch.equal(self.batch(mode="off")["inputs_embeds"], self.batch(mode="text")["inputs_embeds"]))
        self.assertFalse(torch.equal(self.batch()["inputs_embeds"], self.batch(mode="text")["inputs_embeds"]))

    def test_only_top1_tokens_modified(self):
        nn.init.normal_(self.net.output.weight)
        _, selected = encode_prompt(FakeLegacy, self.tokenizer, row())
        diff = (self.batch()["inputs_embeds"] != self.batch(mode="off")["inputs_embeds"]).any(-1)[0]
        self.assertEqual(diff.nonzero().flatten().tolist(), selected)

    def test_no_reference_or_oracle_enters_prompt(self):
        prompt, _ = prompt_and_span(FakeLegacy, self.tokenizer, row())
        self.assertNotIn("PRIVATE_", prompt)
        changed = row(); changed["reference"] = "ANOTHER_REFERENCE"
        self.assertTrue(torch.equal(self.batch()["inputs_embeds"], self.batch(changed)["inputs_embeds"]))

    def test_target_mask_and_no_target_in_connector_query(self):
        seen = []
        hook = self.net.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        a = self.batch(training=True)
        changed = row(); changed["reference"] = "DIFFERENT_TARGET"
        self.batch(changed, training=True)
        hook.remove()
        self.assertTrue(torch.equal(seen[0], seen[1]))
        prefix_len = self.batch()["inputs_embeds"].shape[1]
        self.assertTrue((a["labels"][0, :prefix_len] == -100).all())
        self.assertTrue((a["labels"][0, prefix_len:] >= 0).all())
        self.assertEqual(a["labels"].shape, a["attention_mask"].shape)

    def test_json_escaping_and_duplicate_top1(self):
        example = row('引号"反斜线\\换行\n记者')
        prompt, (a, b) = prompt_and_span(FakeLegacy, self.tokenizer, example)
        self.assertEqual(prompt[a:b], json.dumps(example["input"]["asr_top1"], ensure_ascii=False)[1:-1])
        ids, selected = encode_prompt(FakeLegacy, self.tokenizer, example)
        self.assertEqual(len(selected), b - a)
        self.assertGreater(len(ids), len(selected))

    def test_gradient_reaches_attention_after_first_update(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model(**self.batch(training=True)).loss
            loss.backward()
            self.assertGreater(float(self.net.output.weight.grad.abs().sum()), 0)
            upstream = float(self.net.cross_attention.in_proj_weight.grad.abs().sum())
            if step == 0:
                self.assertEqual(upstream, 0)
            else:
                self.assertGreater(upstream, 0)
                self.assertGreater(float(self.net.audio_projection[1].weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))
            optimizer.step()

    def test_real_audio_changes_residual_after_nonzero_projection(self):
        nn.init.normal_(self.net.output.weight, std=.01)
        x = torch.randn(5, 16)
        self.assertFalse(torch.allclose(self.net(x, self.frames), self.net(x, self.frames.flip(0))))
        self.assertFalse(torch.allclose(self.net(x, self.frames), self.net(x, torch.zeros_like(self.frames))))

    def test_save_reload_exact(self):
        nn.init.normal_(self.net.output.weight, std=.01)
        stream = io.BytesIO(); torch.save(self.net.state_dict(), stream); stream.seek(0)
        other = TextAcousticResidual(self.net.config)
        other.load_state_dict(torch.load(stream, weights_only=True))
        x = torch.randn(5, 16)
        self.assertTrue(torch.equal(self.net(x, self.frames), other(x, self.frames)))

    def test_bfloat16_dtype_and_gradient(self):
        x = torch.randn(5, 16).bfloat16()
        self.assertEqual(self.net(x, self.frames).dtype, torch.bfloat16)
        self.net(x, self.frames).float().sum().backward()
        self.assertIsNotNone(self.net.output.weight.grad)

    def test_empty_top1_rejected(self):
        with self.assertRaises(ValueError):
            encode_prompt(FakeLegacy, self.tokenizer, row(""))

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            audio_path(SimpleNamespace(RUN=Path(".")), "../secret")

    def test_changed_source_rejected_before_loading_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.py"; source.write_text("changed")
            (root / "protocol.json").write_text(json.dumps({"sources": {str(source): "wrong_hash"}}))
            with self.assertRaisesRegex(ValueError, "Locked source changed"):
                verify(root)

    def test_edit_diagnostics(self):
        self.assertEqual(edit_counts("abc", "adc"), {"S": 1, "D": 0, "I": 0})
        self.assertEqual(edit_counts("abc", "ac"), {"S": 0, "D": 1, "I": 0})
        self.assertEqual(edit_counts("abc", "abxc"), {"S": 0, "D": 0, "I": 1})
        self.assertEqual(edit_counts("", "abc")["I"], 3)
        self.assertEqual(edit_counts("abc", "")["D"], 3)
        value = diagnose([dict(reference="abc", prediction="adc", input=dict(asr_top1="abc"))], lambda x: x)
        self.assertEqual(value["correct_input_broken_sentences"], 1)
        self.assertEqual(value["errors_on_correct_inputs"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
