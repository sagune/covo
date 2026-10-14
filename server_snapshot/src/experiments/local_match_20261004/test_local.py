import copy
import unittest
import torch
from local_inputs import aligned_characters,candidates,label_slots,remap_span
from local_connector import LocalMatchConnector,matching_loss

class Tests(unittest.TestCase):
    def test_alignment(self):
        self.assertEqual(aligned_characters('甲乙','甲丙乙'),[0,2])
        self.assertEqual(aligned_characters('甲乙','甲'),[0,None])

    def test_no_reference_in_inputs_and_same_phone_group(self):
        vocab=['<blank>','qiao','jia','wei']
        row=dict(id='x',split='train',reference='乔加伟',input=dict(asr_top1='乔佳伟',nbest=['乔加伟','乔嘉伟']))
        record=candidates(row,vocab); altered=copy.deepcopy(row);altered['reference']='其他答案'
        self.assertEqual(record,candidates(altered,vocab))
        self.assertEqual(len(record['slots'][1]['groups']),1)
        self.assertEqual(set(record['slots'][1]['groups'][0]['aliases']),set('佳加嘉'))
        self.assertEqual(label_slots(row,record),[1,1,1])
        altered['split']='dev'
        with self.assertRaises(AssertionError):label_slots(altered,record)

    def test_bottleneck_and_gradients(self):
        torch.manual_seed(917)
        model=LocalMatchConnector(3,torch.randn(2,16))
        record=dict(frames=8,slots=[dict(span=[0,4],groups=[dict(phone='jia',aliases=['加','佳'])]),dict(span=[4,8],groups=[dict(phone='wei',aliases=['伟'])])])
        vectors,logits=model(torch.randn(8,512),record,dict(jia=1,wei=2))
        self.assertEqual(vectors.shape,(2,16))
        loss=vectors.square().sum()+matching_loss(logits,[1,0]);loss.backward()
        self.assertGreater(float(model.matcher[-1].weight.grad.norm()),0)
        self.assertGreater(float(model.query.weight.grad.norm()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_neutral_same_layout(self):
        model=LocalMatchConnector(2,torch.randn(2,8))
        record=dict(frames=8,slots=[dict(span=[0,8],groups=[dict(phone='jia',aliases=['加'])])])
        a,_=model(torch.randn(8,512),record,dict(jia=1),True)
        b,_=model(torch.zeros(8,512),record,dict(jia=1),True)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(remap_span([6,8],8,3),(2,3))

    def test_mask_and_balanced_loss(self):
        x=torch.tensor([0.,0.,0.],requires_grad=True)
        self.assertAlmostEqual(float(matching_loss(x,[1,0,-1])),.69314718,places=6)
        matching_loss(x,[-1,-1,-1]).backward();self.assertEqual(float(x.grad.sum()),0)

if __name__=='__main__':unittest.main()
