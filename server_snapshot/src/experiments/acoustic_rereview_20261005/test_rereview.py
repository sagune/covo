import sys
from pathlib import Path
import unittest

# Local tests use the same previous evidence source; remote directory auto-resolves it.
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'acoustic-review-20261005'))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'text-residual-coupling-20260920'))
from rereview import compose,parse,seedbit,whitelist


class Tests(unittest.TestCase):
    def test_whitelist(self):
        inp=dict(asr_top1='123',nbest=['123'],hotwords=[dict(text='2')],
                 reference='SECRET',keyword_mentions=['SECRET'],oracle_hotwords=['SECRET'])
        self.assertNotIn('SECRET',str(whitelist(inp)))

    def test_order(self):
        self.assertEqual(seedbit('x:0'),seedbit('x:0'))
        self.assertEqual({seedbit('x:'+str(i)) for i in range(10)},{0,1})

    def test_parse(self):
        q=dict(A=dict(text='2',sentence='123',role='original'),B=dict(text='4',sentence='143',role='modified'))
        self.assertEqual(parse('{"text":"2"}',q)['selected_role'],'original')
        self.assertEqual(parse('{"text":"143"}',q)['selected_role'],'modified')
        self.assertTrue(parse('{"text":"证据不足"}',q)['abstained'])
        self.assertFalse(parse('{"text":"自由改写"}',q)['valid'])
        self.assertFalse(parse('not json',q)['valid'])

    def test_compose(self):
        q=dict(normalized_output='10395',edits=[dict(kind='different_phone_substitution',edit_id='x:0',output_span=[1,2],original='2',modified='0'),
             dict(kind='different_phone_substitution',edit_id='x:1',output_span=[3,4],original='4',modified='9')])
        ds=[dict(edit_id='x:0',decision=dict(valid=True,selected_role='original')),dict(edit_id='x:1',decision=dict(valid=True,selected_role='original'))]
        self.assertEqual(compose(q,ds),('12345',True))
        ds[0]['decision']['valid']=False
        self.assertEqual(compose(q,ds),('',False))
        self.assertEqual(compose(dict(normalized_output='123',edits=[]),[]),('123',True))


if __name__=='__main__':unittest.main()
