"""Focused CPU tests. Run with the experiment's required Python environment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import experiment as e


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id='a', reference='甲', input={'asr_top1':'乙'}),
                     dict(id='b', reference='乙', input={'asr_top1':'甲'})]
        self.permutation = {'a':'b', 'b':'a'}

    def predictions(self, mode):
        return [dict(r, condition=mode, audio_id=self.permutation[r['id']] if mode=='mismatch' else r['id']) for r in self.rows]

    def test_complete_and_partial_identity(self):
        for mode in ('text','true','zero','mismatch'):
            e.check_predictions(self.predictions(mode),self.rows,mode,self.permutation)
            e.check_predictions(self.predictions(mode)[:1],self.rows,mode,self.permutation)

    def test_reject_wrong_id_order_reference_evidence_condition_audio(self):
        for key, value in [('id','bad'),('reference','坏'),('input',{}),('condition','text'),('audio_id','a')]:
            rr=self.predictions('mismatch'); rr[0][key]=value
            with self.assertRaises(AssertionError): e.check_predictions(rr,self.rows,'mismatch',self.permutation)
        with self.assertRaises(AssertionError):
            e.check_predictions(self.predictions('true')[::-1],self.rows,'true',self.permutation)
        with self.assertRaises(AssertionError):
            e.check_predictions(self.predictions('true')*2,self.rows,'true',self.permutation)

    def test_checkpoint_hashes_group_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); f=path/'tensor.txt'; f.write_text('original')
            meta=dict(complete=True,group='B',protocol_sha256='abc',hashes={'tensor.txt':e.w.sha(f)})
            e.w.save(path/'metadata.json',meta)
            self.assertEqual(e.check_checkpoint(path,{'protocol_sha256':'abc'},'B'),meta)
            with self.assertRaises(AssertionError): e.check_checkpoint(path,{'protocol_sha256':'abc'},'C')
            with self.assertRaises(AssertionError): e.check_checkpoint(path,{'protocol_sha256':'def'},'B')
            f.write_text('changed')
            with self.assertRaises(AssertionError): e.check_checkpoint(path,{'protocol_sha256':'abc'},'B')

    def test_optimizers_exact_parameter_sets_and_schedule(self):
        import torch
        for group in ('A','B','C'):
            rt=e.Runtime.__new__(e.Runtime); rt.group=group
            rt.c=dict(lr=5e-5,lora_lr=5e-6,weight_decay=.01,warmup=32,steps=2162)
            rt.connector=torch.nn.Linear(2,2)
            lora=torch.nn.Parameter(torch.ones(2)); rt.lora=[] if group=='A' else [('lora_A',lora)]
            opt,sch=rt.optimizer()
            actual={id(x) for g in opt.param_groups for x in g['params']}
            expected=({id(lora)} if group!='A' else set()) | ({id(x) for x in rt.connector.parameters()} if group in ('A','B') else set())
            self.assertEqual(actual,expected)
            self.assertAlmostEqual(opt.param_groups[-1]['lr'],(5e-5 if group=='A' else 5e-6)/32)
            for _ in range(2162): opt.step(); sch.step()
            self.assertEqual(sch.last_epoch,2162)
            self.assertTrue(all(g['lr']==0 for g in opt.param_groups))

    def test_final_accumulation_retains_six_rows(self):
        rows=list(range(17294)); chunks=[rows[i*8:(i+1)*8] for i in range(2162)]
        self.assertEqual(len(chunks[-1]),6)
        self.assertEqual([x for chunk in chunks for x in chunk],rows)


if __name__=='__main__': unittest.main()
