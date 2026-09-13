import argparse
import copy
import importlib.util
import itertools
import unittest
from pathlib import Path

import cbsensevoice_covo_bridge as bridge


def args(**overrides):
    parser = argparse.ArgumentParser()
    bridge.add_prepare_args(parser)
    result = parser.parse_args(['--input', 'unused', '--output', 'unused'])
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class BridgeBTests(unittest.TestCase):
    def setUp(self):
        self.record = {'reference': '秘密答案', 'input': {
            'asr_top1': '北京天气很好', 'nbest': ['北京天气很好', '北京天气真好'],
            'hotwords': [{'text': '北京', 'score': .9}, {'text': '朱圣祎', 'score': .2},
                        {'text': '朱圣祎', 'score': .1}, {'text': '另一个词'}],
            'prompt_hotwords': [{'text': '北京', 'weight': 1}],
            'keyword_mentions': [{'mention': '不能传入'}], 'oracle_hotwords': [{'text': '金标准词'}]}}

    def test_modes_and_source_filters(self):
        before = copy.deepcopy(self.record)
        for compact, structured, source in itertools.product([True, False], [True, False], ['prompt', 'kws', 'all']):
            options = args(compact_evidence=compact, include_hotword_evidence=structured,
                           hotword_source=source, max_hotwords=1, include_missing_kws_hotwords=True)
            audit = {}
            output = bridge.build_user_prompt(self.record, options, audit)
            for word in ['北京', '朱圣祎', '另一个词']:
                self.assertIn(word, output)
            for word in ['秘密答案', '不能传入', '金标准词']:
                self.assertNotIn(word, output)
            self.assertEqual(audit['missing_after'], [])
            self.assertEqual(audit['recorded_unique'], 3)
            self.assertEqual(audit['added'].count('朱圣祎'), 1)
        self.assertEqual(self.record, before)

    def test_only_appends(self):
        old = bridge.build_user_prompt(self.record, args())
        new = bridge.build_user_prompt(self.record, args(include_missing_kws_hotwords=True))
        extra, _ = bridge.supplement_missing_kws_hotwords(self.record['input'], old.splitlines()[:-1])
        self.assertEqual(new, '\n'.join(old.splitlines()[:-1] + extra + old.splitlines()[-1:]))

    def test_empty_and_already_visible(self):
        for data in [{}, {'hotwords': [{'text': ''}]}, {'hotwords': [{'text': '北京'}]}]:
            extra, audit = bridge.supplement_missing_kws_hotwords(data, ['北京天气很好'])
            self.assertEqual(extra, [])
            self.assertEqual(audit['missing_after'], [])

    def test_no_cross_candidate_match(self):
        extra, audit = bridge.supplement_missing_kws_hotwords({'hotwords': [{'text': '北京'}]}, ['河北', '京城'])
        self.assertEqual(audit['added'], ['北京'])
        self.assertTrue(extra)

    def test_reference_independence(self):
        options = args(include_missing_kws_hotwords=True)
        output = bridge.build_user_prompt(self.record, options)
        altered = copy.deepcopy(self.record)
        altered['reference'] = '另一个答案'
        altered['input']['keyword_mentions'] = []
        altered['input']['oracle_hotwords'] = []
        self.assertEqual(output, bridge.build_user_prompt(altered, options))

    def test_legacy_module_equivalence(self):
        path = Path(__file__).with_name('cbsensevoice_covo_bridge.original.py')
        if not path.exists():
            self.skipTest('Original snapshot unavailable')
        spec = importlib.util.spec_from_file_location('legacy_bridge', path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for compact, structured, source in itertools.product([True, False], [True, False], ['prompt', 'kws', 'all']):
            options = args(compact_evidence=compact, include_hotword_evidence=structured, hotword_source=source)
            self.assertEqual(bridge.build_user_prompt(self.record, options), legacy.build_user_prompt(self.record, options))


if __name__ == '__main__':
    unittest.main()
