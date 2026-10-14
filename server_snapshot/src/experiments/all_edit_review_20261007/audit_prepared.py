"""Independent all-pair preparation checks; no model or GPU loading."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path('/root/autodl-tmp/src/experiments')
sys.path.insert(0, str(ROOT/'acoustic_review_20261005'))
from review import norm, align
sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface import render, swap, order_bit


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def audit(here):
    c,d=load(here/'protocol.json'),load(here/'data_lock.json')
    assert d['protocol_sha256']==sha(here/'protocol.json')
    for path,digest in {**c['sources'],**d['hashes']}.items():assert sha(path)==digest,path
    assert c['tasks']==['full'] and c['full_weight']==1. and c['local_weight']==0.
    assert (d['pairs'],d['updates'],d['draft_failures'])==(18956,2370,49)
    assert c['normalization']=='NFKC/remove whitespace/Unicode punctuation, NOOpenCC'
    assert not any((here/a).exists() for a in ('AUDIO','TEXT','UNADAPTED'))
    tr,dv=rows(c['train']),rows(c['dev']);train={x['id']:x for x in tr}
    old=rows(c['baseline']);drafts=rows(here/'train_drafts.jsonl')
    assert [x['id'] for x in tr]==[x['id'] for x in drafts]
    for fn in (lambda x:x['id'],lambda x:x['id'].split('W')[0],lambda x:x['wav']):
        assert not {fn(x) for x in tr}&{fn(x) for x in dv}
    qs,ls=rows(here/'train_queries.jsonl'),rows(here/'train_labels.jsonl')
    dq=rows(here/'dev_queries.jsonl');man=rows(here/'train_edit_manifest.jsonl')
    assert len(qs)==len(ls)==18956 and len(dq)==604
    assert [q['edit_id'] for q in qs]==[l['edit_id'] for l in ls]
    zby={z['edit_id']:(m,z) for m in man for z in m['edits']}
    assert set(zby)=={q['edit_id'] for q in qs}
    assert dict(Counter(z['kind'] for m in man for z in m['edits']))==c['train_type_counts']
    for q,l in zip(qs,ls):
        assert set(q)=={'id','edit_id','output_span','draft','input_evidence','A','B','evidence'}
        assert set(q['input_evidence'])=={'asr_top1','nbest','hotwords'}
        m,z=zby[q['edit_id']];assert q['draft']==m['normalized_output'] and q['output_span']==z['output_span']
        candidates={q[k]['role']:q[k] for k in ('A','B')}
        assert candidates['original']['text']==z['original'] and candidates['modified']['text']==z['modified']
        assert candidates['original']['sentence']==z['candidate_original_context'] and candidates['modified']['sentence']==z['candidate_modified_context']
        ref=norm(train[q['id']]['reference']);er={k:distance(ref,norm(q[k]['sentence'])) for k in ('A','B')}
        prob=1. if er['A']<er['B'] else 0. if er['A']>er['B'] else .5
        assert l['errors']==er and l['prob_A']==prob and l['neutral']==(prob==.5)
        clean=render(q,'true');dirty=dict(q,reference='REFERENCE_SECRET',diagnosis='REFERENCE_SECRET',prob_A=prob)
        assert render(dirty,'true')==clean and 'REFERENCE_SECRET' not in json.dumps(clean)
    assert [r['id'] for r in old]==[r['id'] for r in dv]
    allblocks=[]
    for row in old:
        # Independently extract all nonmatch blocks (never read the reference).
        a,b=norm(row['input']['asr_top1']),norm(row['prediction']);pending=[];i=0
        for op in align(a,b)[1]+[('M',len(a),len(b),'','')]:
            if op[0]!='M':pending.append(op);continue
            if not pending:continue
            ai,bi=pending[0][1:3];aa=''.join(x[3] for x in pending);bb=''.join(x[4] for x in pending)
            allblocks.append((row['id']+':'+str(i),[bi,bi+len(bb)],aa,bb));pending=[];i+=1
    assert [(q['edit_id'],q['output_span'],next(q[k]['text'] for k in ('A','B') if q[k]['role']=='original'),
             next(q[k]['text'] for k in ('A','B') if q[k]['role']=='modified')) for q in dq]==allblocks
    ids=[q['edit_id'] for q in qs];random.Random(917).shuffle(ids);assert ids==d['train_order']
    # Verify all encoded model inputs against the locked whitelisted prompts.
    import torch
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
    bank=torch.load(here/'token_bank.pt',map_location='cpu',weights_only=True)
    by={q['edit_id']:q for q in qs+dq};dev_ids={q['edit_id'] for q in dq}
    assert set(by)=={key[0] for key in bank}
    for i,(key,tensor) in enumerate(bank.items()):
        uid,role,mode,task=key;assert task=='full'
        q=by[uid];qq=q if q['A']['role']==role else swap(q)
        expected=tok(tok.apply_chat_template(render(qq,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False,return_tensors='pt').input_ids
        assert torch.equal(tensor,expected) and tensor.shape[0]==1 and 0<tensor.shape[1]<=c['max_sequence']
        if (i+1)%5000==0:print(json.dumps({'audited_token_inputs':i+1,'total':len(bank)}),flush=True)
    for q in qs:
        qq=swap(q) if order_bit(q['edit_id']) else q
        assert all((q['edit_id'],qq['A']['role'],m,'full') in bank for m in ('true','none'))
    assert dev_ids.isdisjoint(train)
    assert all((q['edit_id'],q['A']['role'],m,'full') in bank for q in dq for m in ('none','true','zero','mismatch'))
    out=dict(status='PASS',independent_train_labels=True,all_dev_routes_reference_free=True,all_token_inputs_verified=True,
             pairs=18956,updates=2370,dev_pairs=604,token_inputs=len(bank),old_best_initialization=c['checkpoint'],
             initial_adapter_sha256=sha(Path(c['checkpoint'])/'adapter/adapter_model.safetensors'),
             protocol_sha256=sha(here/'protocol.json'),data_lock_sha256=sha(here/'data_lock.json'),no_formal_training_started=True)
    with (here/'preparation_independent_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    print(json.dumps(out),flush=True)


if __name__=='__main__':audit(Path(sys.argv[1]))
