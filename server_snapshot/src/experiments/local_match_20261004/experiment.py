"""Matched CE versus CE+local phonetic compatibility; no output rules."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'phonetic_reader_20261004'
sys.path.insert(0,str(PARENT))
import reader
import fast
from local_inputs import candidates,label_slots,timing,norm,pronunciation,aligned_characters
from local_connector import LocalMatchConnector,matching_loss
w=reader.w
ARMS={'CE':0.,'MATCH':.2}
MODES=['true','neutral','zero','mismatch']

def parts(p,row,record):
    system='你是中文语音转写助手。仅输出JSON对象，格式为{"text":"最终转写"}，不输出解释。'
    user=('请校正语音识别结果，不要仅为流畅改写。局部发音支持仅说明候选读音与声音的匹配程度，'
          '不是汉字答案；相同读音的写法结合上下文和热词判断。\n识别证据：'
          +json.dumps(p.evidence(row),ensure_ascii=False)+'\n按识别文本字位置编号的局部候选与发音支持：\n')
    index=0
    for slot in record['slots']:
        user+=f"位置{slot['position']+1}，原字{slot['base']}："
        if not slot['groups']:user+='无已知发音候选。\n';continue
        for g in slot['groups']:
            user+=f"{'/'.join(g['aliases'])}（{g['phone']}），支持：@@LOCAL_{index}@@；"
            index+=1
        user+='\n'
    return [dict(role='system',content=system),dict(role='user',content=user)],index

def prepare():
    from transformers import AutoTokenizer
    assert not (HERE/'protocol.json').exists(),'Locked protocol already exists'
    parent,p=reader.configuration();fast.read_optimization()
    rows=p.load(parent['train']);dev=p.load(parent['dev']);vocab=w.read(parent['vocabulary'])
    assert (len(rows),len(dev))==(17294,1334)
    assert not ({r['id'] for r in rows}&{r['id'] for r in dev})
    assert all(r['split']=='train' for r in rows) and all(r['split']=='dev' for r in dev)
    assert not ({norm(r['reference']) for r in rows}&{norm(r['reference']) for r in dev})
    head=reader.probe.make_model('linear',len(vocab)).eval()
    head.load_state_dict(torch.load(Path(parent['phone_checkpoint'])/'state.pt',map_location='cpu',weights_only=False)['model'])
    cache=w.read(parent['cache_manifest']);torch.set_num_threads(6)
    tok=AutoTokenizer.from_pretrained(p.BASE,trust_remote_code=True)
    state_ids={word:tok(word,add_special_tokens=False)['input_ids'] for word in ['否','是']}
    assert all(len(x)==1 for x in state_ids.values()),state_ids
    manifests=[];labels=[];counts=dict(fallback=0,unanchored_slots=0,queries=0,positive=0,negative=0,unlabelled=0)
    longest=(0,None);began=time.time()
    for i,row in enumerate(rows+dev):
        record=candidates(row,vocab)
        # No reference is accessed for candidates, timing or the decoder inputs.
        altered=dict(row,reference='REFERENCE_MUST_NOT_AFFECT_INPUTS')
        assert record==candidates(altered,vocab)
        info=cache[row['id']];assert w.sha(info['path'])==info['sha256']
        frames=torch.from_numpy(np.load(info['path'],allow_pickle=False).copy()).float()
        assert len(frames)==info['frames'] and torch.isfinite(frames).all()
        with torch.no_grad():logits=head(frames)
        timing(record,len(frames),logits,vocab,reader.forced_phone_midpoints)
        counts['fallback']+=record['timing_fallback'] is not None
        counts['unanchored_slots']+=sum(not s['timing_anchored'] for s in record['slots'])
        msgs,n=parts(p,row,record);counts['queries']+=n
        prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        for k in range(n):prompt=prompt.replace(f'@@LOCAL_{k}@@','')
        bound=len(tok(prompt,add_special_tokens=False)['input_ids'])+n
        if row['split']=='train':
            ll=label_slots(row,record);assert len(ll)==n
            labels.append(dict(id=row['id'],labels=ll))
            for value in ll:counts[{1:'positive',0:'negative',-1:'unlabelled'}[value]]+=1
            bound+=len(tok(json.dumps({'text':row['reference']},ensure_ascii=False)+tok.eos_token,add_special_tokens=False)['input_ids'])
            if bound>longest[0]:longest=(bound,row['id'])
        assert bound+2*n<=4096,('Conservative sequence bound',row['id'],bound,n)
        manifests.append(record)
        if (i+1)%1000==0:print(json.dumps(dict(prepared=i+1,seconds=time.time()-began,**counts)),flush=True)
    p.save_rows(HERE/'input_slots.jsonl',manifests);p.save_rows(HERE/'train_match_labels.jsonl',labels)
    p.save_rows(HERE/'empty_auxiliary.jsonl',[])
    names=['experiment.py','local_inputs.py','local_connector.py','test_local.py','PLAN.md','input_slots.jsonl','train_match_labels.jsonl','empty_auxiliary.jsonl']
    c={**parent,'version':2,'parent_protocol_sha256':parent['protocol_sha256'],
       'sources':{str((HERE/name).resolve()):w.sha(HERE/name) for name in names},
       'auxiliary':str(HERE/'empty_auxiliary.jsonl'),'auxiliary_n':0,'auxiliary_empty':0,
       'input_slots':str(HERE/'input_slots.jsonl'),'match_labels':str(HERE/'train_match_labels.jsonl'),
       'arms':ARMS,'modes':MODES,'state_token_ids':state_ids,'preparation_counts':counts,
       'smoke_ids':list(dict.fromkeys([rows[0]['id'],longest[1],next(r['id'] for r in labels if 0 in r['labels'] and 1 in r['labels'])])),
       'microbatch':4,'objective':'CE: final JSON CE; MATCH: same CE plus0.2 balanced local candidate-phone BCE per example',
       'interface':'input-only ASR-character positions; top1 plus first8 Nbest; maximum4 unique phonetic groups per position; spelling aliases share one query/value',
       'timing':'frozen CTC Viterbi on INPUT top1 phones only; common slot windows across candidates; infeasible input alignment uniform fallback, retain row',
       'semantics':'learned phonetic query attends512D local frames; same compatibility logits feed BCE and decoder;2-state soft value only, no raw acoustic prefix',
       'frozen':'Base9B and old best B/text LoRA; frozen linear phone probe used only for CPU input timing',
       'trainable':'same new rank8 evidence LoRA, phonetic-query matcher and2-state compatibility embeddings in BOTH arms',
       'selection':'Fixed final2162 updates,17294 rows once, same seed/order/initial tensor state; no dev checkpoint selection',
       'limitations':['Historical1334 contextual dev, not official14326 dev or new heldout test',
         'Input-top1 CTC timing can be wrong, including uniform fallback; boundaries are weak, not listened gold',
         'Character slots, not word lattice; Nbest insertions absent as new slots; final decoder remains open-vocabulary',
         'TRAIN labels align references to input positions and use dictionary toneless phones; deleted/nonphonetic slots unlabelled',
         'Phonetic groups limited to first4 from first8 input Nbest; original Nbest still supplied to9B; no oracle candidate injection',
         'Timing kept identical for zero/neutral; mismatch rescales the same input timing to permuted cache duration',
         'Neutral is learned state midpoint with identical prompt/layout, not exact old text route',
         '9B still may choose wrong same-phone spellings; shared audio value is not a CER guarantee',
         'Both arms use pre-existing audited FLA audio operators; text uses historical fallback',
         'Main CE may train compatibility via language preferences; matched loss comparison is the locked hypothesis test']}
    c.pop('protocol_sha256',None)
    c['sources'][str((PARENT/'protocol.json').resolve())]=w.sha(PARENT/'protocol.json')
    c['sources'][str((PARENT/'optimization.json').resolve())]=w.sha(PARENT/'optimization.json')
    baseline=reader.old.HERE/'B/dev/text.predictions.jsonl'
    baseline_rows=p.load(baseline)
    reader.old.check_predictions(baseline_rows,dev,'text',parent['audio_permutation'])
    assert len(baseline_rows)==1334
    c['sources'][str(baseline.resolve())]=w.sha(baseline)
    w.save(HERE/'protocol.json',c)
    for arm in ARMS:(HERE/arm).mkdir(exist_ok=False)
    w.save(HERE/'status.json',dict(status='PREPARED',train=17294,dev=1334,counts=counts))

def configuration():
    parent,p=reader.configuration();fast.read_optimization();c=w.read(HERE/'protocol.json')
    assert c['parent_protocol_sha256']==parent['protocol_sha256']
    for path,expected in c['sources'].items():assert w.sha(path)==expected,('Locked source changed',path)
    assert w.sha(c['cache_manifest'])==c['cache_manifest_sha256']
    c['protocol_sha256']=w.sha(HERE/'protocol.json');return c,p

class Runtime(fast.FastRuntime):
    def __init__(self,c,p,training,checkpoint=None):
        super().__init__(c,p,training,checkpoint=None)
        embed=self.model.get_input_embeddings()
        with torch.no_grad():states=torch.stack([embed.weight[c['state_token_ids'][word][0]].float() for word in ['否','是']])
        self.connector=LocalMatchConnector(len(w.read(c['vocabulary'])),states).cuda().train(training)
        self.phone_ids={x:i for i,x in enumerate(w.read(c['vocabulary']))}
        self.records={r['id']:r for r in p.load(c['input_slots'])}
        self.labels={r['id']:r['labels'] for r in p.load(c['match_labels'])}
        self.kernel_enabled=True;self.microbatch=4
        self.parameters=list(self.evidence.values())+list(self.connector.parameters())
        if checkpoint:self.load(checkpoint)

    def batch_local(self,row,mode='true',training=False):
        self.activate(True);record=self.records[row['id']]
        uid=self.c['audio_permutation'][row['id']] if mode=='mismatch' else row['id']
        frames=torch.from_numpy(np.load(self.cache[uid]['path'],allow_pickle=False).copy()).float().cuda()
        if mode=='zero':frames=torch.zeros_like(frames)
        values,logits=self.connector(frames,record,self.phone_ids,mode=='neutral')
        msgs,n=parts(self.p,row,record);assert n==len(values)
        prompt=self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        embed=self.model.get_input_embeddings();pieces=[]
        def text(s):return embed(torch.tensor(self.tokenizer(s,add_special_tokens=False)['input_ids'],device='cuda'))
        for i in range(n):
            before,prompt=prompt.split(f'@@LOCAL_{i}@@',1);pieces.extend([text(before),values[i:i+1].to(embed.weight.dtype)])
        pieces.append(text(prompt));length=sum(map(len,pieces));labels=None
        if training:
            assert row['split']=='train'
            target=json.dumps({'text':row['reference']},ensure_ascii=False)+self.tokenizer.eos_token
            ids=torch.tensor(self.tokenizer(target,add_special_tokens=False)['input_ids'],device='cuda')
            labels=torch.cat([torch.full((length,),-100,device='cuda',dtype=torch.long),ids])[None];pieces.append(embed(ids))
        x=torch.cat(pieces)[None];assert x.shape[1]<=4096,row['id']
        result=dict(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long))
        if labels is not None:result['labels']=labels
        loss=matching_loss(logits,self.labels[row['id']]) if training else None
        return result,loss,logits

    def generate_local(self,row,mode):
        with torch.no_grad():
            batch,_,logits=self.batch_local(row,mode)
            ids=self.model.generate(**batch,do_sample=False,max_new_tokens=96,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        assert ids.shape[1]<=96
        raw=self.tokenizer.decode(ids[0],skip_special_tokens=True);prediction,parsed=self.p.parse_prediction(raw)
        return dict(id=row['id'],reference=row['reference'],input=row['input'],prediction=prediction,parsed=parsed,
          model_output=raw,condition=mode,audio_id=self.c['audio_permutation'][row['id']] if mode=='mismatch' else row['id'],
          compatibility_logits=logits.float().cpu().tolist())

def init_hash(rt):
    h=hashlib.sha256()
    items={**{f'lm:{k}':v for k,v in rt.evidence.items()},**{f'connector:{k}':v for k,v in rt.connector.state_dict().items()}}
    for name,value in sorted(items.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def update(rt,chunk,opt,sch,weight):
    opt.zero_grad(set_to_none=True);ces=[];matches=[]
    for start in range(0,len(chunk),4):
        rr=chunk[start:start+4];built=[rt.batch_local(r,training=True) for r in rr]
        x,y=fast.collate([b[0] for b in built]);ce=fast.target_loss(rt.model,x,y)
        ml=torch.stack([b[1] for b in built]);loss=(ce.sum()+weight*ml.sum())/len(chunk)
        assert torch.isfinite(loss);loss.backward();ces.extend(ce.detach().cpu().tolist());matches.extend(ml.detach().cpu().tolist())
    assert all(x.grad is None for n,x in rt.model.named_parameters() if '.evidence.' not in n)
    reader_grad=sum(float(x.grad.float().norm()) for x in rt.evidence.values() if x.grad is not None)
    connector_grad=sum(float(x.grad.float().norm()) for x in rt.connector.parameters() if x.grad is not None)
    matcher_grad=float(rt.connector.matcher[-1].weight.grad.norm())
    assert reader_grad>0 and connector_grad>0 and matcher_grad>0
    grad=torch.nn.utils.clip_grad_norm_(rt.parameters,1.);assert torch.isfinite(grad)
    opt.step();sch.step()
    return dict(correction_ce=sum(ces)/len(ces),matching_bce=sum(matches)/len(matches),match_weight=weight,
        weighted_objective=(sum(ces)+weight*sum(matches))/len(chunk),reader_grad=reader_grad,connector_grad=connector_grad,
        matcher_grad=matcher_grad,grad_norm=float(grad),max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)

def check_checkpoint(path,c,arm):
    meta=reader.check_checkpoint(path,c);assert meta['arm']==arm;return meta

def checkpoint(rt,opt,sch,path,step,examples,arm):
    reader.checkpoint(rt,opt,sch,path,step,examples)
    meta=w.read(path/'metadata.json');meta.update(arm=arm,match_weight=ARMS[arm],initial_tensor_sha256=w.read(HERE/'initialization.json')['tensor_sha256'])
    w.save(path/'metadata.json',meta)

def smoke(arm,reload=False):
    c,p=configuration();d=HERE/arm;byid={r['id']:r for r in p.load(c['train'])};cp=d/'smoke_checkpoint'
    with w.gpu_guard(p):
        rt=Runtime(c,p,True);opt,sch=rt.optimizer();identity=init_hash(rt)
        if (HERE/'initialization.json').exists():assert identity==w.read(HERE/'initialization.json')['tensor_sha256']
        else:w.save(HERE/'initialization.json',dict(tensor_sha256=identity,seed=917))
        rt.model.eval();rt.connector.eval();row=byid[c['smoke_ids'][0]]
        baseline=rt.generate(row,'text')['model_output']
        if reload:
            reader.restore(rt,opt,sch,cp);meta=check_checkpoint(cp,c,arm)
            assert sch.last_epoch==meta['step'] and all(int(v['step'])==meta['step'] for v in opt.state.values())
            state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(v,rt.evidence[n].cpu()) for n,v in state['evidence'].items())
            assert all(torch.equal(v,rt.connector.state_dict()[n].cpu()) for n,v in state['connector'].items())
            assert rt.generate_local(row,'true')['model_output']==w.read(d/'smoke_before_reload.json')['model_output']
            assert baseline==rt.generate(row,'text')['model_output']
            rt.model.train();rt.connector.train();rec=update(rt,[row],opt,sch,ARMS[arm])
            w.save(d/'smoke.json',dict(status='PASS',reload_exact=True,text_unchanged=True,protocol_sha256=c['protocol_sha256'],initial_tensor_sha256=identity,resumed_update=rec))
        else:
            assert not cp.exists()
            frozen={n:x._version for n,x in rt.model.named_parameters() if '.evidence.' not in n}
            rt.model.train();rt.connector.train()
            for uid in c['smoke_ids']:
                began=time.time();rec=update(rt,[byid[uid]]*4,opt,sch,ARMS[arm])
                assert all(x._version==frozen[n] for n,x in rt.model.named_parameters() if n in frozen)
                assert rec['max_gpu_gb']<30
                print(json.dumps(dict(smoke_id=uid,seconds=time.time()-began,**rec)),flush=True)
            checkpoint(rt,opt,sch,cp,len(c['smoke_ids']),len(c['smoke_ids'])*4,arm)
            rt.model.eval();rt.connector.eval();assert baseline==rt.generate(row,'text')['model_output']
            a=rt.batch_local(row,'neutral')[0];b=rt.batch_local(row,'neutral')[0]
            assert torch.equal(a['inputs_embeds'],b['inputs_embeds'])
            w.save(d/'smoke_before_reload.json',rt.generate_local(row,'true'))

def train(arm):
    c,p=configuration();d=HERE/arm;assert w.read(d/'smoke.json')['protocol_sha256']==c['protocol_sha256']
    if (d/'selected_checkpoint.json').exists():
        meta=check_checkpoint(Path(w.read(d/'selected_checkpoint.json')['checkpoint']),c,arm)
        assert (meta['step'],meta['examples'])==(2162,17294);return
    (d/'checkpoints').mkdir(exist_ok=True)
    completed=[(check_checkpoint(q,c,arm)['step'],q) for q in (d/'checkpoints').glob('step_*') if (q/'metadata.json').exists()]
    start,cp=max(completed,default=(0,None))
    with w.gpu_guard(p):
        rt=Runtime(c,p,True);opt,sch=rt.optimizer()
        if cp:reader.restore(rt,opt,sch,cp)
        else:assert init_hash(rt)==w.read(HERE/'initialization.json')['tensor_sha256']
        rows=p.load(c['train']);random.Random(917).shuffle(rows);began=time.time()
        frozen={n:x._version for n,x in rt.model.named_parameters() if '.evidence.' not in n}
        with (d/f'training_{time.time_ns()}.jsonl').open('x') as stream:
            for step in range(start,2162):
                rec=update(rt,rows[step*8:(step+1)*8],opt,sch,ARMS[arm])
                assert all(x._version==frozen[n] for n,x in rt.model.named_parameters() if n in frozen)
                rec.update(status='TRAINING',arm=arm,step=step+1,steps=2162,examples=min((step+1)*8,17294),seconds=time.time()-began,resume_step=start)
                stream.write(json.dumps(rec)+'\n');stream.flush();w.save(d/'status.json',rec)
                if step==start or (step+1)%10==0:print(json.dumps(rec),flush=True)
                if (step+1)%256==0 or step+1==2162:
                    q=d/'checkpoints'/f'step_{step+1}_{time.time_ns()}';checkpoint(rt,opt,sch,q,step+1,rec['examples'],arm)
                    w.save(d/'latest_checkpoint.json',dict(checkpoint=str(q),step=step+1))
        w.save(d/'selected_checkpoint.json',dict(checkpoint=str(q),step=2162,reason=c['selection']))
        w.save(d/'status.json',dict(status='TRAINED_NOT_EVALUATED'))

def evaluate(arm):
    c,p=configuration();d=HERE/arm;cp=Path(w.read(d/'selected_checkpoint.json')['checkpoint'])
    meta=check_checkpoint(cp,c,arm);assert (meta['step'],meta['examples'])==(2162,17294)
    rows=p.load(c['dev']);dest=d/'dev';dest.mkdir(exist_ok=True)
    missing=[m for m in MODES if not (dest/f'{m}.predictions.jsonl').exists()]
    if not missing:return
    with w.gpu_guard(p):
        rt=Runtime(c,p,False,cp)
        for mode in missing:
            target=dest/f'{mode}.predictions.jsonl';partial=Path(str(target)+'.partial')
            prior=p.load(partial) if partial.exists() else [];reader.old.check_predictions(prior,rows,mode,c['audio_permutation'])
            with partial.open('a',encoding='utf-8') as stream:
                for i in range(len(prior),len(rows)):
                    pred=rt.generate_local(rows[i],mode);stream.write(json.dumps(pred,ensure_ascii=False)+'\n');stream.flush()
                    if (i+1)%25==0 or i+1==1334:
                        rec=dict(status='EVALUATING',arm=arm,mode=mode,done=i+1,total=1334);w.save(d/'status.json',rec);print(json.dumps(rec),flush=True)
            rr=p.load(partial);reader.old.check_predictions(rr,rows,mode,c['audio_permutation']);assert len(rr)==1334;partial.rename(target)

def compatibility_diagnostic(records,slots):
    counts=dict(valid=0,positive=0,negative=0,correct=0,pair_slots=0,pair_correct=0,pair_ties=0,unknown=0)
    for row in records:
        rec=slots[row['id']]
        # Retrospective dev labels, never used in construction or selection.
        lab=label_slots(dict(row,split='train'),rec);scores=row['compatibility_logits'];assert len(lab)==len(scores)
        offset=0
        for slot in rec['slots']:
            n=len(slot['groups']);ll=lab[offset:offset+n];ss=scores[offset:offset+n];offset+=n
            pos=[s for l,s in zip(ll,ss) if l==1];neg=[s for l,s in zip(ll,ss) if l==0]
            for l,s in zip(ll,ss):
                if l<0:counts['unknown']+=1;continue
                counts['valid']+=1;counts['positive' if l else 'negative']+=1;counts['correct']+=(s>=0)==bool(l)
            if pos and neg:
                counts['pair_slots']+=1;counts['pair_correct']+=max(pos)>max(neg);counts['pair_ties']+=max(pos)==max(neg)
    counts['warning']='Reference/dictionary retrospective slot labels; not wave-verified or deployable selection'
    return counts

def report():
    c,p=configuration();rows=p.load(c['dev']);slots={r['id']:r for r in p.load(c['input_slots'])}
    baseline=p.load(reader.old.HERE/'B/dev/text.predictions.jsonl')
    reader.old.check_predictions(baseline,rows,'text',c['audio_permutation']);assert len(baseline)==1334
    metrics={'old_text':reader.old.score(p,baseline)};diagnostics={};hashes={'old_text':w.sha(reader.old.HERE/'B/dev/text.predictions.jsonl')}
    for arm in ARMS:
        for mode in MODES:
            name=f'{arm}_{mode}';path=HERE/arm/'dev'/f'{mode}.predictions.jsonl';rr=p.load(path)
            reader.old.check_predictions(rr,rows,mode,c['audio_permutation']);assert len(rr)==1334
            metrics[name]=reader.old.score(p,rr);w.save(path.with_name(mode+'.metrics.json'),metrics[name])
            diagnostics[name]=compatibility_diagnostic(rr,slots);hashes[name]=w.sha(path)
    comparisons={name:p.bootstrap(metrics['MATCH_true']['per_utterance'],v['per_utterance']) for name,v in metrics.items() if name!='MATCH_true'}
    result=dict(protocol_sha256=c['protocol_sha256'],summaries={k:v['summary'] for k,v in metrics.items()},
       error_diagnostics={k:v['error_diagnostics'] for k,v in metrics.items()},compatibility=diagnostics,
       MATCH_true_minus_controls_95ci=comparisons,prediction_hashes=hashes,initial_tensor_sha256=w.read(HERE/'initialization.json')['tensor_sha256'],
       no_protection=True,limitations=c['limitations'])
    w.save(HERE/'summary.json',result);w.save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

def pipeline():
    import fcntl
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[('smoke',a) for a in ARMS]+[('reload-smoke',a) for a in ARMS]+[('train',a) for a in ARMS]+[('eval',a) for a in ARMS]+[('report',None)]
        for command,arm in jobs:
            if command in ('smoke','reload-smoke') and (HERE/arm/'smoke.json').exists():continue
            if command=='smoke' and (HERE/arm/'smoke_checkpoint/metadata.json').exists():continue
            w.save(HERE/'status.json',dict(status='RUNNING',phase=command,arm=arm,pid=os.getpid(),updated=time.time()))
            args=[sys.executable,'-u',str(Path(__file__)),command]+([arm] if arm else [])
            with (HERE/f'{command}_{arm}_{time.time_ns()}.log').open('x') as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','reload-smoke','train','eval','report','pipeline']);parser.add_argument('arm',choices=list(ARMS),nargs='?');args=parser.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='pipeline':pipeline()
        elif args.command=='report':report()
        else:
            assert args.arm
            if args.command=='smoke':smoke(args.arm)
            elif args.command=='reload-smoke':smoke(args.arm,True)
            elif args.command=='train':train(args.arm)
            else:evaluate(args.arm)
    except BaseException:
        error=dict(status='FAILED',command=args.command,arm=args.arm,error=traceback.format_exc(),time=time.time())
        w.save(HERE/f'failure_{time.time_ns()}.json',error);w.save(HERE/'status.json',error);raise
