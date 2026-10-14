import copy
import json
import unittest
from interface import render,swap,preference,label_for,choose


def query():
    scores={m:{r:{'score':v} for r,v in [('original',-8.),('modified',-2.)]} for m in ('native','phonetic_marginal')}
    return dict(id='train1',edit_id='train1:0',output_span=[0,1],draft='甲',input_evidence={'asr_top1':'乙','nbest':['甲','乙'],'hotwords':[]},
        A={'role':'original','text':'乙','sentence':'乙','pinyin':['yi']},B={'role':'modified','text':'甲','sentence':'甲','pinyin':['jia']},
        evidence={m:copy.deepcopy(scores) for m in ('true','zero','mismatch')})


class Tests(unittest.TestCase):
    def test_reference_not_rendered(self):
        q=query();base=render(q,'true');q.update(reference='SECRET',prob_A=1,keyword_mentions=['SECRET'])
        self.assertEqual(render(q,'true'),base);self.assertNotIn('SECRET',json.dumps(base))
    def test_local_has_no_context_or_roles(self):
        q=query();q['draft']='DRAFT_SECRET';q['input_evidence']['asr_top1']='INPUT_SECRET';q['A']['sentence']='SENTENCE_SECRET'
        text=json.dumps(render(q,'true','local'),ensure_ascii=False)
        for x in ('DRAFT_SECRET','INPUT_SECRET','SENTENCE_SECRET','original','modified'):self.assertNotIn(x,text)
    def test_none_and_missing(self):
        q=query();self.assertNotIn('-8.0',json.dumps(render(q,'none')))
        q['evidence']['true']['native']['original']['score']=None
        self.assertIn('不可评分',json.dumps(render(q,'true'),ensure_ascii=False))
    def test_label_swap(self):
        q=query();f=lambda a,b:int(a!=b);lab=preference(q,'乙',str,f,'train')
        self.assertEqual(lab['prob_A'],1);self.assertEqual(label_for(q,lab,True),0)
        self.assertEqual(preference(swap(q),'乙',str,f,'train')['prob_A'],0)
    def test_neutral_and_split(self):
        q=query();self.assertEqual(preference(q,'丙',str,lambda a,b:int(a!=b),'train')['prob_A'],.5)
        with self.assertRaises(ValueError):preference(q,'乙',str,lambda a,b:0,'dev')
    def test_selection(self):
        self.assertEqual(choose(1,0),'A');self.assertEqual(choose(0,1),'B');self.assertEqual(choose(1,1),'A')
        with self.assertRaises(ValueError):choose(float('nan'),0)


if __name__=='__main__':unittest.main()
