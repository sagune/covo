"""Frozen old B/text + new pronunciation reader; no inference scoring/rules."""
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

HERE=Path(__file__).resolve().parent
PROBE=HERE.parent/'phonetic_probe_20261004'
sys.path.insert(0,str(PROBE))
import probe
old=probe.prior; w=probe.w
sys.path.insert(0,str(HERE))
from connector import PronunciationConnector,forced_phone_midpoints

def integer(uid,tag):
    return int(hashlib.sha256(('917|'+tag+'|'+uid).encode()).hexdigest()[:16],16)

def prepare():
    import numpy as np
    import torch
    assert not (HERE/'protocol.json').exists(), 'Prepared sources are immutable'
    c,p=old.configuration(); pc,_=probe.configuration()
    selected=w.read(old.HERE/'B/selected_checkpoint.json'); base=Path(selected['checkpoint'])
    old.check_checkpoint(base,c,'B')
    ps=w.read(PROBE/'linear/selected_checkpoint.json'); phone=Path(ps['checkpoint'])
    pm=w.read(phone/'metadata.json')
    assert (pm['epoch'],pm['updates'],pm['protocol_sha256'])==(10,5410,pc['protocol_sha256'])
    assert w.sha(phone/'state.pt')==pm['state_sha256']
    vocab=w.read(pc['vocabulary']); ids={x:i for i,x in enumerate(vocab)}
    cache=w.read(c['cache_manifest']); train=p.load(c['train']); dev=p.load(c['dev'])
    targets={r['id']:r for r in p.load(pc['train_targets'])}
    assert (len(train),len(dev),len(targets))==(17294,1334,17294)
    assert set(targets)=={r['id'] for r in train}
    torch.set_num_threads(6)
    head=probe.make_model('linear',len(vocab)).eval()
    head.load_state_dict(torch.load(phone/'state.pt',map_location='cpu',weights_only=False)['model'])
    auxiliaries=[]; began=time.time()
    for row in train:
        assert row['split']=='train'
        if integer(row['id'],'aux_select')%4: continue
        frames=torch.from_numpy(np.load(cache[row['id']]['path'],allow_pickle=False).copy()).float()
        width=min(24,len(frames)); start=integer(row['id'],'aux_window')%(len(frames)-width+1)
        end=start+width; phones=targets[row['id']]['syllables']
        with torch.no_grad(): log_probs=head(frames).log_softmax(-1)
        midpoints=forced_phone_midpoints(log_probs,[ids[x] for x in phones])
        label=[x for x,mid in zip(phones,midpoints) if start<=mid<end]
        auxiliaries.append(dict(id=row['id'],start=start,end=end,pinyin=label))
        if len(auxiliaries)%500==0: print(json.dumps(dict(prepared_aux=len(auxiliaries),seconds=time.time()-began)),flush=True)
    p.save_rows(HERE/'train_local_phone_labels.jsonl',auxiliaries)
    # CPU token-length audit uses a conservative one-token-per-frame bound.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p.BASE,trust_remote_code=True)
    longest=(0,None)
    for row in train+dev:
        prompt=tok.apply_chat_template(messages(p,row,True),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        before,after=prompt.split(p.MARKER)
        count=sum(len(tok(s,add_special_tokens=False)['input_ids']) for s in (before,after))+cache[row['id']]['frames']
        if row['split']=='train': count+=len(tok(json.dumps({'text':row['reference']},ensure_ascii=False)+tok.eos_token,add_special_tokens=False)['input_ids'])
        assert count<=4096,('Conservative length bound',row['id'],count)
        if row['split']=='train': longest=max(longest,(count,row['id']))
    paths=[HERE/name for name in ('reader.py','connector.py','test_reader.py','PLAN.md','train_local_phone_labels.jsonl')]
    paths += [old.HERE/'B/selected_checkpoint.json',base/'metadata.json',
              base/'adapter/adapter_config.json',base/'adapter/adapter_model.safetensors',
              phone/'state.pt',phone/'metadata.json',PROBE/'protocol.json',Path(pc['vocabulary'])]
    protocol=dict(version=1,parent_protocol_sha256=c['protocol_sha256'],probe_protocol_sha256=pc['protocol_sha256'],
        sources={str(q.resolve()):w.sha(q) for q in paths},train=c['train'],dev=c['dev'],
        cache_manifest=c['cache_manifest'],cache_manifest_sha256=w.sha(c['cache_manifest']),
        vocabulary=pc['vocabulary'],phone_checkpoint=str(phone),old_text_checkpoint=str(base),
        auxiliary=str(HERE/'train_local_phone_labels.jsonl'),seed=917,train_n=17294,dev_n=1334,
        steps=2162,accumulation=8,connector_lr=5e-5,reader_lr=5e-6,weight_decay=.01,warmup=32,
        checkpoint_interval=256,auxiliary_weight=.2,auxiliary_n=len(auxiliaries),
        auxiliary_empty=sum(not r['pinyin'] for r in auxiliaries),max_sequence=4096,max_new_tokens=96,
        audio_permutation=c['audio_permutation'],smoke_ids=list(dict.fromkeys([train[0]['id'],longest[1],next(r['id'] for r in auxiliaries if r['pinyin'])])),
        reader_lora=dict(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),
        modes=['text','true','zero','mismatch','no_phonetic','frozen_reader'],
        objective='Correction JSON CE on all train +0.2 local audio-only pinyin JSON CE on preselected train quarter',
        frozen='Base9B, old B/text LoRA, linear phone head, pinyin vocabulary and 9B-derived pronunciation anchor embeddings',
        trainable='New audio-only rank8 evidence LoRA + acoustic projection + pronunciation gain; no old adapter updates',
        segmentation='Audio posterior argmax runs capped16 frames; all frames including blank retained; soft posteriors not hard transcripts',
        semantics='Posterior-weighted old9B input embeddings of spaced pinyin spellings; blank anchor0; continuous512D local acoustic residual retained',
        auxiliary_labels='24-frame hash-chosen TRAIN windows; train-reference CTC Viterbi midpoint phones as weak labels only. Window choice independent of reference. Empty targets retained.',
        selection='Fixed final2162 updates,17294 rows exactly once; no dev checkpoint selection or tuning',
        no_protection=True,limitations=[*pc['limitations'],
            'This first version has audio-local segmentation and linguistic anchors, NOT explicit text-token forced time alignment',
            'Local weak timestamps derive from train reference/dictionary and CTC, not manually annotated boundaries',
            'Composite architecture+prompt+auxiliary change, not a causal training-objective ablation',
            'no_phonetic removes soft semantic values, but retains phonetic-head-derived segmentation',
            'Audio-only evidence adapter freezes old text route, but does not guarantee audio-route CER non-degradation',
            'Zero erases raw cached frames and resegments; duration/segmentation counts differ from true'])
    w.save(HERE/'protocol.json',protocol); (HERE/'R').mkdir(exist_ok=False)
    w.save(HERE/'status.json',dict(status='PREPARED',auxiliary_n=len(auxiliaries),auxiliary_empty=protocol['auxiliary_empty']))

def configuration():
    parent,p=old.configuration(); c=w.read(HERE/'protocol.json')
    assert parent['protocol_sha256']==c['parent_protocol_sha256']
    for name,expected in c['sources'].items(): assert w.sha(name)==expected,('Locked source changed',name)
    assert w.sha(c['cache_manifest'])==c['cache_manifest_sha256']
    c['protocol_sha256']=w.sha(HERE/'protocol.json')
    return c,p

def messages(p,row,acoustic):
    if not acoustic: return p.messages(row,'B',False)
    system='你是中文语音转写助手。仅输出JSON对象，格式为{"text":"最终转写"}，不输出解释。'
    user=('请校正语音识别结果。以下连续表示按语音时间顺序提供局部发音依据，包含不确定的拼音含义与声学细节，'
          '不是另一份确定的转写答案。请把发音与识别文本对应起来，根据实际说话内容补充或修正，'
          '不要仅为流畅改写。同音词的具体写法结合上下文和热词判断，热词不必强行加入。\n局部发音依据：'
          +p.MARKER+'\n识别证据：'+json.dumps(p.evidence(row),ensure_ascii=False))
    return [dict(role='system',content=system),dict(role='user',content=user)]

class Runtime:
    def __init__(self,c,p,training,checkpoint=None):
        import torch
        from peft import LoraConfig
        self.c,self.p,self.training=c,p,training
        self.model,self.tokenizer,unused=p.load_model('B',Path(c['old_text_checkpoint']),False)
        del unused
        self.model.add_adapter('evidence',LoraConfig(**c['reader_lora']))
        vocab=w.read(c['vocabulary']); self.cache=w.read(c['cache_manifest'])
        self.aux={r['id']:r for r in p.load(c['auxiliary'])}
        phone=probe.make_model('linear',len(vocab))
        phone.load_state_dict(torch.load(Path(c['phone_checkpoint'])/'state.pt',map_location='cpu',weights_only=False)['model'])
        embed=self.model.get_input_embeddings(); anchors=[]
        with torch.no_grad():
            for value in vocab:
                if value=='<blank>': anchors.append(torch.zeros(embed.weight.shape[1],device='cuda')); continue
                ids=self.tokenizer(' '+value+' ',add_special_tokens=False)['input_ids']
                anchors.append(embed(torch.tensor(ids,device='cuda')).float().mean(0))
        self.connector=PronunciationConnector(phone,torch.stack(anchors),embed.weight.shape[1]).cuda()
        self.activate(True)
        if training:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.config.use_cache=False; self.model.train()
            for module in self.model.modules():
                if isinstance(module,torch.nn.Dropout): module.p=0.
        else: self.model.eval()
        self.connector.train(training)
        self.evidence={n:x for n,x in self.model.named_parameters() if '.evidence.' in n}
        assert self.evidence and all('lora_' in n for n in self.evidence)
        self.parameters=list(self.evidence.values())+[x for x in self.connector.parameters() if x.requires_grad]
        if checkpoint: self.load(checkpoint)

    def activate(self,audio,reader=True):
        # PeftModel.set_adapter accepts one name; inspected LoraModel API supports
        # the additive pair. Wrapper active_adapter remains valid old 'default'.
        self.model.base_model.set_adapter(['default','evidence'] if audio and reader else ['default'],inference_mode=True)
        for name,param in self.model.named_parameters():
            param.requires_grad_(self.training and '.evidence.' in name and audio and reader)

    def batch(self,row,mode='true',training=False,auxiliary=False):
        import numpy as np
        import torch
        p=self.p; audio=mode!='text'; self.activate(audio,mode!='frozen_reader')
        if auxiliary:
            assert training and row['split']=='train' and row['id'] in self.aux
            msgs=[dict(role='system',content='你是语音发音助手。仅输出JSON对象，格式为{"pinyin":["音节"]}，用不带声调的拼音，不输出汉字或解释。'),
                  dict(role='user',content='请读出这个局部语音片段实际包含的拼音音节，按时间顺序列出：'+p.MARKER)]
        else: msgs=messages(p,row,audio)
        prompt=self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        embed=self.model.get_input_embeddings()
        def text(value): return embed(torch.tensor(self.tokenizer(value,add_special_tokens=False)['input_ids'],device='cuda'))
        if audio:
            before,after=prompt.split(p.MARKER)
            uid=self.c['audio_permutation'][row['id']] if mode=='mismatch' else row['id']
            frames=torch.from_numpy(np.load(self.cache[uid]['path'],allow_pickle=False).copy()).float().cuda()
            if auxiliary:
                a=self.aux[row['id']]; frames=frames[a['start']:a['end']]
            if mode=='zero': frames=torch.zeros_like(frames)
            acoustic=self.connector(frames,mode!='no_phonetic').to(embed.weight.dtype)
            parts=[text(before),acoustic,text(after)]
        else: parts=[text(prompt)]
        length=sum(map(len,parts)); labels=None
        if training:
            # Reference is accessed only for the target AFTER evidence construction.
            value={'pinyin':self.aux[row['id']]['pinyin']} if auxiliary else {'text':row['reference']}
            target=json.dumps(value,ensure_ascii=False)+self.tokenizer.eos_token
            ids=torch.tensor(self.tokenizer(target,add_special_tokens=False)['input_ids'],device='cuda')
            labels=torch.cat([torch.full((length,),-100,device='cuda',dtype=torch.long),ids])[None]
            parts.append(embed(ids))
        inputs=torch.cat(parts)[None]; assert inputs.shape[1]<=self.c['max_sequence'],row['id']
        result=dict(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],device='cuda',dtype=torch.long))
        if labels is not None: result['labels']=labels
        return result

    def optimizer(self):
        import torch
        c=self.c
        opt=torch.optim.AdamW([dict(params=list(self.evidence.values()),lr=c['reader_lr']),
                              dict(params=[p for p in self.connector.parameters() if p.requires_grad],lr=c['connector_lr'])],weight_decay=c['weight_decay'])
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/c['warmup'] if s<c['warmup'] else max(0.,(c['steps']-s)/(c['steps']-c['warmup'])))
        return opt,sch

    def load(self,path):
        import torch
        meta=check_checkpoint(path,self.c)
        state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
        assert state['evidence'].keys()==self.evidence.keys()
        with torch.no_grad():
            for name,param in self.evidence.items(): param.copy_(state['evidence'][name])
        self.connector.load_state_dict(state['connector'])
        return state,meta

    def generate(self,row,mode):
        import torch
        with torch.no_grad():
            batch=self.batch(row,mode)
            ids=self.model.generate(**batch,do_sample=False,max_new_tokens=self.c['max_new_tokens'],
                                    pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        assert ids.shape[1]<=self.c['max_new_tokens']
        raw=self.tokenizer.decode(ids[0],skip_special_tokens=True); pred,parsed=self.p.parse_prediction(raw)
        return dict(id=row['id'],reference=row['reference'],input=row['input'],prediction=pred,parsed=parsed,
                    model_output=raw,condition=mode,audio_id=self.c['audio_permutation'][row['id']] if mode=='mismatch' else row['id'])

def update(rt,chunk,opt,sch):
    import torch
    opt.zero_grad(set_to_none=True); ce=[]; auxiliary=[]
    for row in chunk:
        loss=rt.model(**rt.batch(row,training=True),use_cache=False).loss
        assert torch.isfinite(loss); (loss/len(chunk)).backward(); ce.append(float(loss.detach()))
        if row['id'] in rt.aux:
            loss=rt.model(**rt.batch(row,training=True,auxiliary=True),use_cache=False).loss
            assert torch.isfinite(loss); (rt.c['auxiliary_weight']*loss/len(chunk)).backward(); auxiliary.append(float(loss.detach()))
    assert all(x.grad is None for n,x in rt.model.named_parameters() if '.evidence.' not in n)
    reader_grad=sum(float(x.grad.float().norm()) for x in rt.evidence.values() if x.grad is not None)
    connector_grad=sum(float(x.grad.float().norm()) for x in rt.connector.parameters() if x.grad is not None)
    assert reader_grad>0 and connector_grad>0 and math.isfinite(reader_grad+connector_grad)
    grad=torch.nn.utils.clip_grad_norm_(rt.parameters,1.); assert torch.isfinite(grad)
    opt.step(); sch.step()
    return dict(correction_ce=sum(ce)/len(ce),local_phone_ce=sum(auxiliary)/len(auxiliary) if auxiliary else None,
                local_phone_n=len(auxiliary),reader_grad=reader_grad,connector_grad=connector_grad,grad_norm=float(grad))

def checkpoint(rt,opt,sch,path,step,examples):
    import numpy as np
    import torch
    path.mkdir(exist_ok=False)
    torch.save(dict(connector=rt.connector.state_dict(),evidence={n:x.detach().cpu() for n,x in rt.evidence.items()},
                    optimizer=opt.state_dict(),scheduler=sch.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step),path/'state.pt')
    w.save(path/'metadata.json',dict(complete=True,step=step,examples=examples,protocol_sha256=rt.c['protocol_sha256'],state_sha256=w.sha(path/'state.pt')))

def check_checkpoint(path,c):
    meta=w.read(path/'metadata.json')
    assert meta['complete'] and meta['protocol_sha256']==c['protocol_sha256'] and meta['state_sha256']==w.sha(path/'state.pt')
    return meta

def restore(rt,opt,sch,path):
    import numpy as np
    import torch
    state,meta=rt.load(path); assert state['step']==meta['step']
    opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    return meta

def smoke(reload=False):
    import torch
    c,p=configuration(); d=HERE/'R'; dest=d/'smoke_checkpoint'; by_id={r['id']:r for r in p.load(c['train'])}
    with w.gpu_guard(p):
        rt=Runtime(c,p,True); opt,sch=rt.optimizer(); n=len(c['smoke_ids'])
        rt.model.eval(); rt.connector.eval()
        baseline=rt.generate(by_id[c['smoke_ids'][0]],'text')
        if reload:
            state,meta=rt.load(dest); restore(rt,opt,sch,dest)
            assert meta['step']==n and sch.last_epoch==n
            assert all(int(v['step'])==n for v in opt.state.values())
            assert all(torch.equal(v,rt.evidence[k].cpu()) for k,v in state['evidence'].items())
            assert all(torch.equal(v,rt.connector.state_dict()[k].cpu()) for k,v in state['connector'].items())
            assert rt.generate(by_id[c['smoke_ids'][0]],'true')['model_output']==w.read(d/'smoke_before_reload.json')['model_output']
            assert baseline['model_output']==rt.generate(by_id[c['smoke_ids'][0]],'text')['model_output']
            rt.model.train(); rt.connector.train(); rec=update(rt,[by_id[c['smoke_ids'][0]]],opt,sch)
            w.save(d/'smoke.json',dict(status='PASS',reload_exact=True,text_route_unchanged=True,resumed_update=rec,
                                      protocol_sha256=c['protocol_sha256'],max_gpu_gb=torch.cuda.max_memory_allocated()/1e9))
        else:
            assert not dest.exists(); frozen={name:x._version for name,x in rt.model.named_parameters() if '.evidence.' not in name}
            before={name:x.detach().cpu().clone() for name,x in rt.evidence.items()}; rt.model.train(); rt.connector.train()
            for uid in c['smoke_ids']:
                began=time.time(); rec=update(rt,[by_id[uid]],opt,sch)
                print(json.dumps(dict(smoke_id=uid,seconds=time.time()-began,**rec)),flush=True)
            assert any(not torch.equal(before[name],x.cpu()) for name,x in rt.evidence.items())
            assert all(x._version==frozen[name] for name,x in rt.model.named_parameters() if name in frozen)
            checkpoint(rt,opt,sch,dest,n,n)
            rt.model.eval(); rt.connector.eval()
            assert baseline['model_output']==rt.generate(by_id[c['smoke_ids'][0]],'text')['model_output']
            w.save(d/'smoke_before_reload.json',rt.generate(by_id[c['smoke_ids'][0]],'true'))
            w.save(d/'smoke_initial.json',dict(status='PASS',frozen_versions_unchanged=True,text_route_unchanged=True,
                                            trainable_reader=sum(x.numel() for x in rt.evidence.values()),max_gpu_gb=torch.cuda.max_memory_allocated()/1e9))

def train():
    import torch
    c,p=configuration(); d=HERE/'R'; assert w.read(d/'smoke.json')['protocol_sha256']==c['protocol_sha256']
    if (d/'selected_checkpoint.json').exists():
        meta=check_checkpoint(Path(w.read(d/'selected_checkpoint.json')['checkpoint']),c)
        assert (meta['step'],meta['examples'])==(2162,17294); return
    complete=[(check_checkpoint(q,c)['step'],q) for q in (d/'checkpoints').glob('step_*') if (q/'metadata.json').exists()]
    start,cp=max(complete,default=(0,None))
    with w.gpu_guard(p):
        rt=Runtime(c,p,True,cp); opt,sch=rt.optimizer(); rows=p.load(c['train']); random.Random(c['seed']).shuffle(rows)
        if cp: restore(rt,opt,sch,cp)
        (d/'checkpoints').mkdir(exist_ok=True); began=time.time(); frozen={n:x._version for n,x in rt.model.named_parameters() if '.evidence.' not in n}
        w.save(d/f'training_started_{time.time_ns()}.json',dict(pid=os.getpid(),start=began,resume_step=start,protocol_sha256=c['protocol_sha256']))
        with (d/f'training_{time.time_ns()}.jsonl').open('x') as stream:
            for step in range(start,c['steps']):
                chunk=rows[step*8:(step+1)*8]; rec=update(rt,chunk,opt,sch)
                assert all(x._version==frozen[n] for n,x in rt.model.named_parameters() if n in frozen)
                rec.update(status='TRAINING',step=step+1,steps=c['steps'],examples=min((step+1)*8,len(rows)),
                           seconds=time.time()-began,resume_step=start,max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
                stream.write(json.dumps(rec)+'\n'); stream.flush(); w.save(d/'status.json',rec)
                if step==start or (step+1)%10==0: print(json.dumps(rec),flush=True)
                if (step+1)%c['checkpoint_interval']==0 or step+1==c['steps']:
                    q=d/'checkpoints'/f'step_{step+1}_{time.time_ns()}'; checkpoint(rt,opt,sch,q,step+1,rec['examples'])
                    w.save(d/'latest_checkpoint.json',dict(checkpoint=str(q),step=step+1))
        w.save(d/'selected_checkpoint.json',dict(checkpoint=str(q),step=2162,reason=c['selection']))
        w.save(d/'status.json',dict(status='TRAINED_NOT_EVALUATED'))

def evaluate():
    c,p=configuration(); d=HERE/'R'; cp=Path(w.read(d/'selected_checkpoint.json')['checkpoint'])
    meta=check_checkpoint(cp,c); assert (meta['step'],meta['examples'])==(2162,17294)
    rows=p.load(c['dev']); dest=d/'dev'; dest.mkdir(exist_ok=True)
    missing=[mode for mode in c['modes'] if not (dest/f'{mode}.predictions.jsonl').exists()]
    if not missing: return
    with w.gpu_guard(p):
        rt=Runtime(c,p,False,cp)
        for mode in missing:
            target=dest/f'{mode}.predictions.jsonl'; partial=Path(str(target)+'.partial')
            prior=p.load(partial) if partial.exists() else []; old.check_predictions(prior,rows,mode,c['audio_permutation'])
            with partial.open('a',encoding='utf-8') as stream:
                for index in range(len(prior),len(rows)):
                    pred=rt.generate(rows[index],mode); stream.write(json.dumps(pred,ensure_ascii=False)+'\n'); stream.flush()
                    if (index+1)%25==0 or index+1==len(rows):
                        rec=dict(status='EVALUATING',mode=mode,done=index+1,total=1334); w.save(d/'status.json',rec); print(json.dumps(rec),flush=True)
            records=p.load(partial); old.check_predictions(records,rows,mode,c['audio_permutation']); assert len(records)==1334
            partial.rename(target)

def report():
    c,p=configuration(); rows=p.load(c['dev']); metrics={}; hashes={}
    for mode in c['modes']:
        path=HERE/'R/dev'/f'{mode}.predictions.jsonl'; rr=p.load(path)
        old.check_predictions(rr,rows,mode,c['audio_permutation']); assert len(rr)==1334
        metrics[mode]=old.score(p,rr); hashes[mode]=w.sha(path); w.save(path.with_name(mode+'.metrics.json'),metrics[mode])
    historic=p.load(old.HERE/'B/dev/text.predictions.jsonl'); current=p.load(HERE/'R/dev/text.predictions.jsonl')
    assert all(a['prediction']==b['prediction'] and a['model_output']==b['model_output'] for a,b in zip(historic,current)), 'Frozen text route differs from historical B: diagnose, do not suppress'
    result=dict(protocol_sha256=c['protocol_sha256'],summaries={k:v['summary'] for k,v in metrics.items()},
                diagnostics={k:v['error_diagnostics'] for k,v in metrics.items()},prediction_hashes=hashes,
                true_minus_controls_95ci={k:p.bootstrap(metrics['true']['per_utterance'],v['per_utterance']) for k,v in metrics.items() if k!='true'},
                no_protection=True,text_route_exactly_matches_old_B=True,limitations=c['limitations'])
    w.save(HERE/'summary.json',result); w.save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

def pipeline():
    import fcntl
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for command in ('smoke','reload-smoke','train','eval','report'):
            if command in ('smoke','reload-smoke') and (HERE/'R/smoke.json').exists(): continue
            if command=='smoke' and (HERE/'R/smoke_checkpoint/metadata.json').exists(): continue
            w.save(HERE/'status.json',dict(status='RUNNING',phase=command,pid=os.getpid(),updated=time.time()))
            with (HERE/f'{command}_{time.time_ns()}.log').open('x') as stream:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),command],stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','smoke','reload-smoke','train','eval','report','pipeline']); args=parser.parse_args()
    try:
        if args.command=='prepare': prepare()
        elif args.command=='smoke': smoke()
        elif args.command=='reload-smoke': smoke(True)
        elif args.command=='train': train()
        elif args.command=='eval': evaluate()
        elif args.command=='report': report()
        else: pipeline()
    except BaseException:
        error=dict(status='FAILED',command=args.command,error=traceback.format_exc(),time=time.time())
        w.save(HERE/f'failure_{time.time_ns()}.json',error); w.save(HERE/'status.json',error); raise
