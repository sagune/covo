"""Independent CPU-only binary choices/output reconstruction/metrics/paired-CI audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'acoustic-review-20261005'))
from review import norm,align

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def score(rr,annotations):
    total=dict(n=len(rr),chars=0,errors=0,input_errors=0,hits=0,mentions=0,lost=0,gained=0,improved=0,worsened=0,unchanged=0,exact=0,empty=0,parse_failures=0)
    diag=dict(S=0,D=0,I=0,input_correct_sentences=0,correct_input_broken_sentences=0,errors_on_correct_inputs=0);per=[]
    for r in rr:
        ref=norm(r['reference']);text=norm(r['prediction']);inp=norm(r['input']['asr_top1']);assert ref
        er,ops=align(ref,text);ie=align(ref,inp)[0];terms=annotations.get(r['id'],[])
        hits=sum(t in text for t in terms)
        for k,v in dict(chars=len(ref),errors=er,input_errors=ie,hits=hits,mentions=len(terms),lost=sum(t in inp and t not in text for t in terms),gained=sum(t not in inp and t in text for t in terms),improved=int(er<ie),worsened=int(er>ie),unchanged=int(er==ie),exact=int(er==0),empty=int(not text),parse_failures=int(not r.get('parsed',True))).items():total[k]+=v
        for op in ops:
            if op[0] in ('S','D','I'):diag[op[0]]+=1
        if ref==inp:
            diag['input_correct_sentences']+=1;diag['correct_input_broken_sentences']+=int(text!=ref);diag['errors_on_correct_inputs']+=er
        per.append(dict(id=r['id'],chars=len(ref),errors=er,hits=hits,mentions=len(terms)))
    total.update(cer=total['errors']/total['chars'],recall=total['hits']/total['mentions'],input_cer=total['input_errors']/total['chars'])
    total['cer_delta_pp']=100*(total['cer']-total['input_cer']);total['relative_cer_change']=(total['errors']-total['input_errors'])/total['input_errors']
    return total,diag,per

def ci(a,b):
    import numpy as np
    assert [x['id'] for x in a]==[x['id'] for x in b]
    # Locked paired utterance resampling; exact seed917/2000 draws.
    rng=np.random.default_rng(917);n=len(a)
    da=np.array([x['errors']-y['errors'] for x,y in zip(a,b)])
    dh=np.array([x['hits']-y['hits'] for x,y in zip(a,b)])
    chars=np.array([x['chars'] for x in a]);mentions=np.array([x['mentions'] for x in a]);cer=[];rec=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n);cer.append(100*da[idx].sum()/chars[idx].sum())
        if mentions[idx].sum():rec.append(100*dh[idx].sum()/mentions[idx].sum())
    return dict(cer_delta_pp_95ci=np.quantile(cer,[.025,.975]).tolist(),recall_delta_pp_95ci=np.quantile(rec,[.025,.975]).tolist())

