import unittest
import torch
from connector import segments,PronunciationConnector,forced_phone_midpoints

class Tests(unittest.TestCase):
    def test_partition_all_frames(self):
        x=torch.tensor([[5.,0.,0.]]*20+[[0.,4.,0.]]*5+[[0.,0.,3.]]*2)
        spans=segments(x)
        self.assertEqual([i for a,b in spans for i in range(a,b)],list(range(27)))
        self.assertTrue(all(b-a<=16 for a,b in spans))
    def test_soft_not_hard(self):
        torch.manual_seed(917); head=torch.nn.Linear(512,3)
        anchors=torch.randn(3,16); anchors[0]=0
        connector=PronunciationConnector(head,anchors,16)
        frames=torch.randn(8,512)
        output=connector(frames); output.square().mean().backward()
        self.assertIsNotNone(connector.pronunciation_gain.grad)
        self.assertGreater(float(connector.acoustic[-1].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in connector.phone_head.parameters()))
        self.assertTrue(torch.equal(connector(frames,False),torch.zeros_like(output)))
    def test_repeated_phone_alignment(self):
        x=torch.tensor([[.01,.99],[.99,.01],[.01,.99]]).log()
        self.assertEqual(forced_phone_midpoints(x,[1,1]),[0.,2.])
    def test_distinct_phone_alignment(self):
        x=torch.tensor([[.01,.98,.01],[.01,.01,.98]]).log()
        self.assertEqual(forced_phone_midpoints(x,[1,2]),[0.,1.])
    def test_too_short_rejected(self):
        with self.assertRaises(AssertionError): forced_phone_midpoints(torch.zeros(2,2),[1,1])
    def test_nan_rejected(self):
        c=PronunciationConnector(torch.nn.Linear(512,3),torch.zeros(3,16),16)
        with self.assertRaises(AssertionError): c(torch.full((2,512),float('nan')))
    def test_additive_reader_preserves_old_route(self):
        from transformers import GPT2Config,GPT2LMHeadModel
        from peft import get_peft_model,LoraConfig
        torch.manual_seed(917)
        config=GPT2Config(vocab_size=32,n_positions=16,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
        model=get_peft_model(GPT2LMHeadModel(config),LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
        ids=torch.tensor([[1,2,3,4]])
        model.eval(); before=model(ids).logits.detach().clone()
        model.add_adapter('evidence',LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
        model.base_model.set_adapter(['default','evidence'],inference_mode=True)
        for n,p in model.named_parameters(): p.requires_grad_('.evidence.' in n)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(ids,labels=ids).loss.backward()
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if '.evidence.' not in n))
        opt.step(); model.base_model.set_adapter(['default'],inference_mode=True)
        self.assertTrue(torch.equal(before,model(ids).logits.detach()))

if __name__=='__main__': unittest.main()
