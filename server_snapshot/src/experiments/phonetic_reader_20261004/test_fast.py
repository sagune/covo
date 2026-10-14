import unittest
from types import SimpleNamespace
import torch
from fast import collate,target_loss

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.projection=torch.nn.Linear(5,11).double()
    def forward(self,inputs_embeds,attention_mask,logits_to_keep,use_cache):
        return SimpleNamespace(logits=self.projection(inputs_embeds[:,logits_to_keep]))

class Tests(unittest.TestCase):
    def test_per_example_not_token_weighting(self):
        torch.manual_seed(917); model=Toy()
        inputs=[torch.randn(1,7,5,dtype=torch.double,requires_grad=True),torch.randn(1,4,5,dtype=torch.double,requires_grad=True)]
        labels=[torch.tensor([[-100,-100,-100,-100,1,2,3]]),torch.tensor([[-100,-100,7,8]])]
        batches=[dict(inputs_embeds=x,labels=y) for x,y in zip(inputs,labels)]
        args,padded=collate(batches); result=target_loss(model,args,padded)
        ref=torch.stack([torch.nn.functional.cross_entropy(model.projection(x[:,:-1]).float().reshape(-1,11),y[:,1:].reshape(-1),ignore_index=-100) for x,y in zip(inputs,labels)])
        torch.testing.assert_close(result,ref,rtol=1e-6,atol=1e-6)
        fast_grad=torch.autograd.grad(result.mean(),[model.projection.weight,*inputs],retain_graph=True)
        ref_grad=torch.autograd.grad(ref.mean(),[model.projection.weight,*inputs])
        for a,b in zip(fast_grad,ref_grad): torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        self.assertEqual(args['attention_mask'].tolist(),[[1]*7,[1]*4+[0]*3])
        self.assertTrue(torch.equal(padded[1,4:],torch.tensor([-100]*3)))
    def test_final_partial_chunk_loss(self):
        # Mean of example means, divided by actual chunk length (including final6).
        ce=torch.tensor([1.,3.,2.,6.,4.,8.]); aux=torch.tensor([2.,5.])
        expected=(ce.sum()+.2*aux.sum())/6
        regrouped=(ce[:4].sum()+ce[4:].sum()+.2*aux.sum())/6
        self.assertEqual(float(expected),float(regrouped))

if __name__=='__main__': unittest.main()
