import unittest
from likelihood import choose, compose, reordered, target, target_positions, single_layout


class Tests(unittest.TestCase):
    def query(self):
        return dict(A=dict(role='original',sentence='记者抵达北京',text='抵',pinyin=['di']),
            B=dict(role='modified',sentence='记者到达北京',text='到',pinyin=['dao']),output_span=[2,3],draft='记者到达北京')

    def test_order(self):
        q=self.query(); swapped=reordered(q,'reversed_order')
        self.assertEqual(swapped['A'],q['B']); self.assertEqual(swapped['B'],q['A'])
        self.assertEqual(reordered(swapped,'reversed_order'),q)

    def test_target(self):
        self.assertEqual(target(self.query(),'A'),'{"text": "记者抵达北京"}')

    def test_causal_index(self):
        self.assertEqual(target_positions(3,7),[3,4,5])
        self.assertEqual(target_positions(3,4),[0,1,2])
        with self.assertRaises(AssertionError):target_positions(4,4)

    def test_selection(self):
        self.assertEqual(choose(-5,-6),'A'); self.assertEqual(choose(-6,-5),'B')
        self.assertEqual(choose(-5,-5),'A')
        with self.assertRaises(AssertionError):choose(float('nan'),-1)

    def test_single_no_padding_layout(self):
        z=single_layout(dict(ids=[1,2,3,4],target_ids=[3,4],prompt_tokens=2))
        self.assertEqual(z,dict(ids=[[1,2,3,4]],mask=[[1,1,1,1]],positions=[[0,1,2,3]],retained=3))
        with self.assertRaises(AssertionError):single_layout(dict(ids=[1,2,3],target_ids=[2,4],prompt_tokens=1))

    def test_compose(self):
        q=self.query()
        self.assertEqual(compose(q['draft'],[(q,'original')]),'记者抵达北京')
        self.assertEqual(compose(q['draft'],[(q,'modified')]),q['draft'])
        self.assertEqual(compose('不变',[]),'不变')
        qq=dict(A=dict(role='original',text='京'),B=dict(role='modified',text='惊'),output_span=[5,6])
        self.assertEqual(compose('记者到达北惊',[(q,'original'),(qq,'original')]),'记者抵达北京')


if __name__=='__main__':unittest.main()
