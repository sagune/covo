"""CPU-only input/reference boundary and edit-location checks."""
import unittest
from review import align, candidates, norm


class Tests(unittest.TestCase):
    def test_alignment(self):
        self.assertEqual(align('记者今天到达北京','记者今天抵达北京')[0],1)
        self.assertEqual(align('今天北京','今天到北京')[0],1)

    def test_normalization(self):
        self.assertEqual(norm('Ａ， B。\n'), 'AB')
        self.assertEqual(norm('臺灣'), '臺灣')

    def test_reference_free(self):
        import inspect
        self.assertEqual(list(inspect.signature(candidates).parameters), ['uid','original','output'])

    def test_reversion_location(self):
        # Numeric-only inputs are retained as explicitly unsupported, not dropped.
        q=candidates('x','12345','10395')
        self.assertEqual(len(q['edits']),2)
        self.assertEqual(q['edits'][0]['candidate_original_context'],'12395')
        self.assertEqual(q['edits'][1]['candidate_original_context'],'10345')
        self.assertTrue(all(e['kind']=='unsupported_pronunciation' for e in q['edits']))

    def test_insertion_deletion(self):
        q=candidates('x','123','1023')
        self.assertEqual(q['edits'][0]['candidate_original_context'],'123')
        q=candidates('x','1023','123')
        self.assertEqual(q['edits'][0]['candidate_original_context'],'1023')

    def test_phones(self):
        try:
            import pypinyin
        except ImportError:
            self.skipTest('pypinyin checked in remote runtime')
        q=candidates('x','今天到达北京','今天抵达北京')
        self.assertEqual(q['edits'][0]['kind'],'different_phone_substitution')
        q=candidates('x','乔加伟','乔佳伟')
        self.assertEqual(q['edits'][0]['kind'],'same_phone_substitution')

    def test_ctc_against_enumeration(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch checked in remote runtime')
        from review import ctc_score
        import itertools
        import math
        probs=torch.tensor([[.3,.4,.3],[.5,.2,.3],[.2,.5,.3]])
        lp=probs.log()
        for target in ([1],[1,1],[1,2],[]):
            total=0.
            for path in itertools.product(range(3),repeat=3):
                collapsed=[];last=None
                for x in path:
                    if x!=last and x!=0:collapsed.append(x)
                    last=x
                if collapsed==target:
                    total+=math.prod(float(probs[t,x]) for t,x in enumerate(path))
            self.assertAlmostEqual(ctc_score(lp,target)['score'],math.log(total),places=5)
        self.assertIsNone(ctc_score(lp,[None])['score'])
        self.assertIsNone(ctc_score(lp,[1,1,1])['score'])


if __name__=='__main__':unittest.main()
