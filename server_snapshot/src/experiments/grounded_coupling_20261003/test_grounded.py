import unittest
from unittest.mock import patch
from types import SimpleNamespace
import grounded as g


class GroundedTests(unittest.TestCase):
    def fake_p(self):
        return SimpleNamespace(evaluator=lambda: SimpleNamespace(norm=lambda s: s),
                               evidence=lambda r: dict(asr_top1=r['input']['asr_top1'], nbest=r['input'].get('nbest', [])))

    def row(self, ref, top1, nbest=None, split='train'):
        return dict(id='a', split=split, reference=ref, input=dict(asr_top1=top1, nbest=nbest or []))

    def test_edit_distance(self):
        for a, b, n in [('甲乙', '甲', 1), ('', '乙', 1), ('甲乙', '乙甲', 2), ('甲', '甲', 0)]:
            self.assertEqual(g.edit_distance(a, b), n)

    def test_no_dev_or_test_mining(self):
        for split in ('dev', 'test'):
            with self.assertRaises(AssertionError):
                g.build_pair(self.row('甲', '乙', split=split), self.fake_p())

    def test_homophones_including_tone_only_excluded(self):
        for ref, wrong in [('杨丙卿', '杨秉清'), ('王惠', '王慧'), ('许', '徐')]:
            pair = g.build_pair(self.row(ref, wrong), self.fake_p())
            self.assertIsNone(pair['negative'])
            self.assertEqual(pair['rejected']['same_phonemes'], 1)

    def test_visible_negative_not_fabricated(self):
        row = self.row('张海澄', '张海棠', ['张海澄', '张海棠', '很长的无关内容'])
        pair = g.build_pair(row, self.fake_p())
        self.assertEqual(pair['negative'], '张海棠')
        self.assertFalse(pair['correct_input'])

    def test_correct_input_and_non_han_guard(self):
        self.assertTrue(g.build_pair(self.row('张海澄', '张海澄'), self.fake_p())['correct_input'])
        self.assertIsNone(g.pronunciation('37岁'))
        self.assertIsNone(g.pronunciation(''))

    def test_deterministic_auxiliary(self):
        self.assertEqual([g.auxiliary_selected(str(i)) for i in range(100)],
                         [g.auxiliary_selected(str(i)) for i in range(100)])

    def test_target_only_causal_scoring_and_gradients(self):
        import torch
        logits = torch.tensor([[[0., 1., 2.], [3., 1., 0.], [0., 0., 0.], [1., 2., 3.]]], requires_grad=True)
        labels = torch.tensor([[-100, -100, 0, 1]])
        ce, score, lp, n = g.target_values(logits, labels)
        expected = torch.stack([logits[0, 1].log_softmax(0)[0], logits[0, 2].log_softmax(0)[1]])
        self.assertTrue(torch.allclose(score, expected.sum()))
        self.assertTrue(torch.allclose(ce, -expected.mean()))
        self.assertEqual(n, 2)
        ce.backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0.)

    def test_contrast_cannot_reward_zero_degradation_by_gradient(self):
        import torch
        a = torch.tensor(0., requires_grad=True); z = torch.tensor(0., requires_grad=True)
        loss = g.contrast_loss(a, z); loss.backward()
        self.assertLess(float(a.grad), 0.)
        self.assertIsNone(z.grad)
        self.assertLess(float(g.contrast_loss(torch.tensor(.5), z)), float(loss))

    def test_peer_kl_zero_if_identical(self):
        import torch
        lp = torch.randn(4, 9).log_softmax(-1)
        loss = torch.nn.functional.kl_div(lp, lp.detach(), log_target=True, reduction='batchmean')
        self.assertAlmostEqual(float(loss), 0., places=6)

    def test_deterministic_scoring_retains_training_checkpointing(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1))
        rt = SimpleNamespace(model=model, connector=torch.nn.Linear(2, 2))
        g.scoring_training_mode(rt, True)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        g.scoring_training_mode(rt, False)
        self.assertTrue(model[1].training)


if __name__ == '__main__':
    unittest.main()
