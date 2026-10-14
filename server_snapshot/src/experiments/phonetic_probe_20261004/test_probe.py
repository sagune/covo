import unittest
import probe as p


class ProbeTests(unittest.TestCase):
    def test_strict_labels_and_homophones(self):
        norm=lambda x:x
        self.assertEqual(p.syllables('杨丙卿',norm),p.syllables('杨秉清',norm))
        self.assertEqual(p.syllables('许',norm),p.syllables('徐',norm))
        self.assertNotEqual(p.syllables('张海澄',norm),p.syllables('张海棠',norm))
        for x in ('37岁','AI',''):
            self.assertIsNone(p.syllables(x,norm))

    def test_ctc_collapse(self):
        vocab=['<blank>','yi','san']
        self.assertEqual(p.greedy([1,1,0,1,2,2,0],vocab),['yi','yi','san'])
        self.assertEqual(p.greedy([0,0],vocab),[])

    def test_repeat_target_minimum(self):
        self.assertEqual(p.min_ctc_frames(['shi','shi','yi']),4)
        self.assertEqual(p.min_ctc_frames(['shi','yi']),2)

    def test_alignment_positions_and_counts(self):
        a=p.align(['yi','qian','san'],['yi','san'])
        self.assertEqual(a['errors'],1); self.assertIsNone(a['positions'][1]); self.assertEqual(a['edits']['D'],1)
        a=p.align(['cheng'],['tang']); self.assertEqual(a['edits']['S'],1)
        a=p.align(['cheng'],['cheng','tang']); self.assertEqual(a['edits']['I'],1)
        self.assertEqual(p.align([],[])['errors'],0)

    def test_bucketing_keeps_all_examples_and_is_deterministic(self):
        rows=[{'id':str(i)} for i in range(17294)]; cache={r['id']:{'frames':i%100+20} for i,r in enumerate(rows)}
        c=dict(seed=917,batch_size=32)
        batches=p.epoch_batches(rows,cache,0,c)
        self.assertEqual(len(batches),541)
        self.assertEqual(len({r['id'] for b in batches for r in b}),17294)
        self.assertEqual(batches,p.epoch_batches(rows,cache,0,c))

    def test_ctc_score_tie_and_unseen(self):
        import torch
        lp=torch.tensor([[0.,4.,0.],[4.,0.,0.],[0.,0.,4.]]).log_softmax(-1)
        ids={'<blank>':0,'yi':1,'san':2}
        good=p.ctc_log_score(lp,['yi','san'],ids)
        self.assertGreater(good,p.ctc_log_score(lp,['san','yi'],ids))
        self.assertEqual(good,p.ctc_log_score(lp,['yi','san'],ids))
        self.assertIsNone(p.ctc_log_score(lp,['qian'],ids))
        self.assertEqual(p.ctc_log_score(lp,[],ids),float(lp[:,0].sum()))

    def test_heads_have_only_probe_parameters_and_gradients(self):
        import torch
        for head in ('linear','mlp'):
            model=p.make_model(head,400)
            x=torch.randn(2,30,512)
            loss=model(x).square().mean(); loss.backward()
            self.assertTrue(all(x.requires_grad for x in model.parameters()))
            self.assertTrue(any(x.grad.abs().sum()>0 for x in model.parameters()))
            self.assertLess(sum(x.numel() for x in model.parameters()),300000)


if __name__=='__main__': unittest.main()
