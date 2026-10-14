import copy
import json
import unittest
from interface import render, swap, preference, label_for, choose


def query():
    scores = {m: {r: {'score': v} for r, v in [('original', -8.), ('modified', -2.)]}
              for m in ('native', 'phonetic_marginal')}
    return dict(id='train1', edit_id='train1:0', output_span=[0, 1], draft='甲',
                input_evidence={'asr_top1': '乙', 'nbest': ['甲', '乙'], 'hotwords': []},
                A={'role': 'original', 'text': '乙', 'sentence': '乙', 'pinyin': ['yi']},
                B={'role': 'modified', 'text': '甲', 'sentence': '甲', 'pinyin': ['jia']},
                evidence={m: copy.deepcopy(scores) for m in ('true', 'zero', 'mismatch')})


class Tests(unittest.TestCase):
    def test_no_reference_serialization(self):
        q = query(); original = render(q, 'true')
        q.update(reference='SECRET', prob_A=1., diagnosis='SECRET', keyword_mentions=['SECRET'])
        self.assertEqual(render(q, 'true'), original)
        self.assertNotIn('SECRET', json.dumps(original))

    def test_full_context_is_retained(self):
        q = query(); text = json.dumps(render(q, 'none'), ensure_ascii=False)
        for key in ('识别证据', '初稿', '完整候选句', 'N-best', '热词', '同音'):
            self.assertIn(key, text)
        with self.assertRaises(AssertionError): render(q, 'true', 'local')

    def test_empty_unknown_and_same_phone(self):
        q = query(); q['A'].update(text='', pinyin=None)
        self.assertIn('空片段', json.dumps(render(q, 'true'), ensure_ascii=False))
        q['A'].update(text='伟', pinyin=['wei']); q['B'].update(text='玮', pinyin=['wei'])
        self.assertIn('不能要求发音分数区别同音汉字', render(q, 'true')[1]['content'])

    def test_none_missing(self):
        q = query(); self.assertNotIn('-8.0', json.dumps(render(q, 'none')))
        q['evidence']['true']['native']['original']['score'] = None
        self.assertIn('不可评分', json.dumps(render(q, 'true'), ensure_ascii=False))

    def test_train_labels_swap(self):
        q = query(); distance = lambda a, b: int(a != b)
        lab = preference(q, '乙', str, distance, 'train')
        self.assertEqual(lab['prob_A'], 1.)
        self.assertEqual(label_for(q, lab, True), 0.)
        self.assertEqual(preference(swap(q), '乙', str, distance, 'train')['prob_A'], 0.)
        self.assertEqual(preference(q, '丙', str, distance, 'train')['prob_A'], .5)
        with self.assertRaises(ValueError): preference(q, '乙', str, distance, 'dev')

    def test_selection(self):
        self.assertEqual(choose(1., 0.), 'A'); self.assertEqual(choose(0., 1.), 'B')
        self.assertEqual(choose(1., 1.), 'A')
        with self.assertRaises(ValueError): choose(float('nan'), 0.)


if __name__ == '__main__': unittest.main()
