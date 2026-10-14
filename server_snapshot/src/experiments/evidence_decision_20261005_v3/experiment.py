"""Frozen SenseVoice evidence; direct9B candidate-decision LoRA, matched controls."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

# Keep numerical execution identical across continuous/reloaded processes.
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ACCEL=Path('/root/autodl-tmp/src/experiments/evidence_decision_runtime_acceleration_20261005/fla052')
sys.path.insert(0,str(ACCEL))

from interface import render,swap,preference,label_for,choose,order_bit

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/src/experiments')
sys.path.insert(0,str(ROOT/'acoustic_rereview_20261005'))
import rereview as r
e=r.evidence;w=r.w
p=w.legacy_module(ROOT/'staged_coupling_20260917')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):return e.sha(path)
def save(path,obj):e.save(path,obj)


def write_rows(path,rr):
    with Path(path).open('x',encoding='utf-8') as f:
        for x in rr:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')


def status(stage,**kw):
    obj=dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    save(HERE/'status.json',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)


def configuration():
    c=load(HERE/'protocol.json')
    for path,expected in c['sources'].items():assert sha(path)==expected,path
    c['protocol_sha256']=sha(HERE/'protocol.json');return c


def prepare():
    assert not (HERE/'protocol.json').exists()
    parent=load(ROOT/'joint_lora_20260923/protocol.json')
    old=load(ROOT/'acoustic_rereview_20261005/protocol.json')
    ds=load(ROOT/'acoustic_rereview_20261005/decision_seal.json')
    assert ds['protocol_sha256']==sha(ROOT/'acoustic_rereview_20261005/protocol.json')
    tr,dv=rows(parent['train']),rows(parent['dev'])
    assert (len(tr),len(dv))==(17294,1334)
    assert all(x['split']=='train' for x in tr) and all(x['split']=='dev' for x in dv)
    tids,dids={x['id'] for x in tr},{x['id'] for x in dv}
    assert len(tids)==17294 and len(dids)==1334 and not tids&dids
    speakers=lambda rr:{x['id'].split('W')[0] for x in rr}
    assert not speakers(tr)&speakers(dv),'Speaker overlap'
    assert not {x['wav'] for x in tr}&{x['wav'] for x in dv}
    cache=load(parent['cache_manifest'])
    assert (tids|dids)<=set(cache)
    status('PREPARING_CACHE_AUDIT',total=len(tids|dids))
    for i,uid in enumerate(sorted(tids|dids)):
        assert sha(cache[uid]['path'])==cache[uid]['sha256'],uid
        if (i+1)%2000==0:status('PREPARING_CACHE_AUDIT',done=i+1,total=len(tids|dids))
    cp=Path(old['checkpoint']);meta=load(cp/'metadata.json')
    assert (meta['group'],meta['step'],meta['examples'])==('B',2162,17294)
    for name in ('adapter/adapter_model.safetensors','adapter/adapter_config.json'):
        assert sha(cp/name)==meta['hashes'][name]
    dq=rows(ROOT/'acoustic_rereview_20261005/queries.jsonl')
    assert len(dq)==190 and len({q['id'] for q in dq})==180
    assert set(q['id'] for q in dq)<=dids
    for q in dq:assert set(q['input_evidence'])=={'asr_top1','nbest','hotwords'}
    with (HERE/'dev_queries.jsonl').open('xb') as f:f.write((ROOT/'acoustic_rereview_20261005/queries.jsonl').read_bytes())
    dep=[HERE/x for x in ('experiment.py','interface.py','test_interface.py','PLAN.md','dev_queries.jsonl')]
    dep += [Path(parent['train']),Path(parent['dev']),Path(parent['cache_manifest']),cp/'metadata.json',cp/'adapter/adapter_model.safetensors',cp/'adapter/adapter_config.json',
        ROOT/'acoustic_review_20261005/review.py',ROOT/'acoustic_rereview_20261005/rereview.py',ROOT/'text_residual_coupling_20260920/workflow.py',ROOT/'text_residual_coupling_20260920/diagnostics.py',
        p.HERE/'run.py',p.RUN/'implementation_lock.json',p.ALIGNED,Path(old['baseline']),ROOT/'acoustic_rereview_20261005/protocol.json',ROOT/'acoustic_rereview_20261005/decision_seal.json',e.SENSE/'model.pt',e.SENSE/'tokens.json',e.SENSE/'config.yaml']
    assert shutil.disk_usage(HERE).free>10*1024**3,'Less than10GiB free'
    c=dict(experiment='9B supervised evidence-grounded candidate decisions',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),sources={str(x):sha(x) for x in dep},
        train=parent['train'],dev=parent['dev'],cache_manifest=parent['cache_manifest'],base=old['base'],checkpoint=old['checkpoint'],baseline=old['baseline'],
        train_n=17294,dev_n=1334,dev_queries=190,reference_chars=21102,hotword_mentions=599,seed=917,
        arms=['AUDIO','TEXT'],tasks=['full','local'],full_weight=.5,local_weight=.5,epochs=1,lr=1e-5,weight_decay=.01,accumulation=8,warmup=32,clip=1.,checkpoint_interval=256,max_sequence=4096,
        choice_contract='Binary next-token A/B soft-target cross-entropy on9B native LM-head; argmax same logits; exacttieA; not free generation',
        train_labels='Only TRAIN reference preference based on normalized complete candidate errors; all neutral pairs soft.5',
        training_controls='Same FULL/LOCAL tasks and candidate order/labels. AUDIO true scores; TEXT unavailable. No zero/mismatch train labels.',
        evaluation_modes=['none','true','zero','mismatch'],evaluate_unadapted=True,reverse_true=True,final_only=True,
        inherited_audio_permutation=parent['audio_permutation'],baseline_replay_ids=old['baseline_replay_ids'],
        normalization='NFKC/remove whitespace/Unicode punctuation, NOOpenCC',
        split_audit=dict(train_ids=17294,dev_ids=1334,id_overlap=0,speaker_overlap=0,wav_overlap=0),
        limitations=['Repeated historical contextual1334 dev, not official14326dev or new heldout','Cached512D float16 native frames; dictionary toneless phones and polyphone/same-phone/orthography issues',
            'NativeCTC contains recognizer prior, not calibrated acoustic confidence; LOCAL hides text context but evidence is whole-utterance CTC',
            'Supervised CE plus local auxiliary encourages but does not prove evidence necessity; no fabricated counterfactual score flips',
            'Generated train drafts are in-sample for initializer; fixed train-only labels, historical error-enriched distribution',
            'Forced candidate decisions not open-vocabulary correction, no claim of raw waveform understanding','Zero/mismatch scores can beOOD; no dev checkpoint/threshold/weight search'])
    save(HERE/'protocol.json',c);status('PREPARED',train_n=17294,dev_n=1334)


def drafts():
    c=configuration();tr=rows(c['train']);final=HERE/'train_drafts.jsonl';part=Path(str(final)+'.partial')
    assert not final.exists() and not part.exists()
    with w.gpu_guard(p):
        rt=r.Runtime(c);began=time.time()
        old={x['id']:x for x in rows(c['baseline'])};replay=[]
        for uid in c['baseline_replay_ids']:
            x=p.generate(rt.model,rt.tok,None,old[uid],'B','text')
            assert x['model_output']==old[uid]['model_output'];replay.append(uid)
        save(HERE/'draft_runtime_audit.json',dict(adapter_exact=True,baseline_raw_replay_ids=replay,no_connector=True,no_trainable_parameters=True))
        with part.open('x',encoding='utf-8') as f:
            for i,row in enumerate(tr):
                # Supply no reference or reference-derived metadata to the generator.
                clean=dict(id=row['id'],reference='',input=r.whitelist(row['input']))
                out=p.generate(rt.model,rt.tok,None,clean,'B','text')
                assert out['reference']==''
                f.write(json.dumps(out,ensure_ascii=False)+'\n');f.flush()
                if (i+1)%25==0 or i+1==len(tr):
                    elapsed=time.time()-began
                    status('FROZEN_TRAIN_DRAFTS',done=i+1,total=len(tr),seconds=elapsed,remaining_seconds_estimate=elapsed/(i+1)*(len(tr)-i-1))
        assert [x['id'] for x in rows(part)]==[x['id'] for x in tr]
        part.rename(final)
    status('TRAIN_DRAFTS_COMPLETE',sha256=sha(final))


def evidence():
    import numpy as np
    import torch
    from funasr import AutoModel
    c=configuration();tr=rows(c['train']);draft=rows(HERE/'train_drafts.jsonl');cache=load(c['cache_manifest'])
    assert [x['id'] for x in tr]==[x['id'] for x in draft]
    manifest=[e.candidates(x['id'],x['input']['asr_top1'],b['prediction']) for x,b in zip(tr,draft)]
    write_rows(HERE/'train_edit_manifest.jsonl',manifest)
    with w.gpu_guard(p),torch.inference_mode():
        torch.set_num_threads(6);wrapper=AutoModel(model=str(e.SENSE),device='cuda:0',disable_update=True)
        model=wrapper.model.eval();tokenizer=wrapper.kwargs['tokenizer'];tokens=load(e.SENSE/'tokens.json')
        assert model.blank_id==0 and len(tokens)==model.ctc.ctc_lo.out_features
        groups={}
        for i,t in enumerate(tokens):
            if len(t)==1 and '\u4e00'<=t<='\u9fff':
                ph=e.phones(t)
                if ph is not None:groups.setdefault(ph[0],[]).append(i)
        pv=['<blank>']+sorted(groups);pi={x:i for i,x in enumerate(pv)}
        gi=[torch.tensor(groups[x],device='cuda',dtype=torch.long) for x in pv[1:]]
        save(HERE/'native_interface.json',dict(native_vocab=len(tokens),phone_vocab=pv,mapping=groups,model_sha256=sha(e.SENSE/'model.pt')))
        output=HERE/'train_evidence.jsonl';part=Path(str(output)+'.partial');assert not output.exists() and not part.exists()
        began=time.time();nq=0
        with part.open('x',encoding='utf-8') as f:
            for i,q in enumerate(manifest):
                edits=[z for z in q['edits'] if z['kind']=='different_phone_substitution'];result=dict(id=q['id'],queries=[])
                if edits:
                    info=cache[q['id']];assert sha(info['path'])==info['sha256']
                    frames=torch.from_numpy(np.load(info['path'],allow_pickle=False).copy()).float().cuda()
                    assert tuple(frames.shape)==(info['frames'],512) and torch.isfinite(frames).all()
                    lp=model.ctc.log_softmax(frames[None])[0];assert torch.isfinite(lp).all()
                    plp=torch.stack([lp[:,0]]+[torch.logsumexp(lp[:,g],dim=-1) for g in gi],dim=-1)
                    plp-=torch.logsumexp(plp,dim=-1,keepdim=True);cached={}
                    def calc(text,ph):
                        if text not in cached:
                            ids=[int(x) for x in tokenizer.encode(text)];bad=any(x==0 or 1<=x<16 or x>=25000 for x in ids)
                            native=dict(score=None,reason='unknown_or_special_native_token') if bad else e.ctc_score(lp,ids)
                            phone=dict(score=None,reason='non_Han_or_unknown_pronunciation') if ph is None else e.ctc_score(plp,[pi.get(x) for x in ph])
                            cached[text]=dict(native=native,phonetic_marginal=phone)
                        return cached[text]
                    for z in edits:
                        a=calc(z['candidate_original_context'],z['original_context_phones']);b=calc(z['candidate_modified_context'],z['modified_context_phones']);scores={}
                        for method in ('native','phonetic_marginal'):
                            aa,bb=a[method],b[method];scores[method]=dict(original=aa,modified=bb,modified_minus_original=bb['score']-aa['score'] if aa['score'] is not None and bb['score'] is not None else None)
                        qq=r.make_query(q,z,tr[i]['input'],{'true':scores})
                        assert set(qq['input_evidence'])=={'asr_top1','nbest','hotwords'}
                        result['queries'].append(qq);nq+=1
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                if (i+1)%100==0 or i+1==len(manifest):status('TRAIN_TRUE_EVIDENCE',done=i+1,total=len(manifest),queries=nq,seconds=time.time()-began)
        assert [x['id'] for x in rows(part)]==[x['id'] for x in tr];part.rename(output)
    save(HERE/'train_evidence_seal.json',dict(protocol_sha256=c['protocol_sha256'],reference_used=False,train_drafts_sha256=sha(HERE/'train_drafts.jsonl'),manifest_sha256=sha(HERE/'train_edit_manifest.jsonl'),evidence_sha256=sha(output)))
    status('TRAIN_EVIDENCE_COMPLETE',queries=nq)


def data_config():
    c=configuration();d=load(HERE/'data_lock.json')
    assert d['protocol_sha256']==c['protocol_sha256']
    for path,digest in d['hashes'].items():assert sha(path)==digest,path
    return c,d


def prepare_reuse():
    """Reuse complete sealed v2 inputs; new backend protocol, never old smoke weights."""
    import torch
    from transformers import AutoTokenizer
    from importlib.metadata import version
    assert not (HERE/'protocol.json').exists()
    parent_dir=ROOT/'evidence_decision_20261005_v2'
    parent=load(parent_dir/'protocol.json');old_d=load(parent_dir/'data_lock.json')
    parent_hash=sha(parent_dir/'protocol.json')
    assert old_d['protocol_sha256']==parent_hash
    for path,digest in parent['sources'].items():assert sha(path)==digest,path
    for path,digest in old_d['hashes'].items():assert sha(path)==digest,path
    seal=load(parent_dir/'train_evidence_seal.json')
    assert seal['protocol_sha256']==parent_hash and not seal['reference_used']
    assert (seal['evidence_sha256'],seal['train_drafts_sha256'])==(sha(parent_dir/'train_evidence.jsonl'),sha(parent_dir/'train_drafts.jsonl'))
    assert (old_d['pairs'],old_d['updates'])==(5325,666)
    assert version('fla-core')==version('flash-linear-attention')=='0.5.2'
    tr,dv=rows(parent['train']),rows(parent['dev'])
    assert (len(tr),len(dv))==(17294,1334)
    assert all(x['split']=='train' for x in tr) and all(x['split']=='dev' for x in dv)
    assert not {x['id'] for x in tr}&{x['id'] for x in dv}
    assert not {x['id'].split('W')[0] for x in tr}&{x['id'].split('W')[0] for x in dv}
    assert not {x['wav'] for x in tr}&{x['wav'] for x in dv}
    files=['dev_queries.jsonl','train_drafts.jsonl','train_edit_manifest.jsonl','train_evidence.jsonl','train_evidence_seal.json','train_queries.jsonl','train_labels.jsonl']
    for name in files:
        dest=HERE/name;assert not dest.exists();dest.symlink_to(parent_dir/name)
        assert sha(dest)==sha(parent_dir/name)
    sources=dict(parent['sources'])
    for path in [parent_dir/'protocol.json',parent_dir/'data_lock.json']+[HERE/x for x in ('experiment.py','interface.py','test_interface.py','PLAN.md')]:sources[str(path)]=sha(path)
    for path in ACCEL.rglob('*'):
        if path.is_file() and (path.suffix in ('.py','.json','.so') or path.name in ('METADATA','RECORD','WHEEL')):sources[str(path)]=sha(path)
    c=copy.deepcopy(parent)
    c.update(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),sources=sources,
        revision='v3 numerical reproducibility and throughput optimization, before formal training',
        parent_protocol=str(parent_dir/'protocol.json'),parent_protocol_sha256=parent_hash,
        parent_data_lock_sha256=sha(parent_dir/'data_lock.json'),reused_parent_evidence_seal=True,
        numerical_settings=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms=True,matmul_tf32=False,cudnn_tf32=False),
        runtime_versions={name:version(name) for name in ('torch','transformers','peft','triton','einops','flash-linear-attention','fla-core')},
        acceleration=dict(path=str(ACCEL),fla_core='0.5.2',flash_linear_attention='0.5.2',
            require_fla_gated_delta=True,causal_conv_backend='original PyTorch fallback',
            gradient_checkpointing=False,pretokenized_reference_free_inputs=True,single_sequence_no_padding=True),
        preparation='Reuse complete hash-verified v2 sealed drafts/evidence/labels/order; no regeneration or dev-based label selection')
    c['limitations'].append('Fused attention/norm kernels can round differently from historical fallback; UNADAPTED A/B controls rerun with the identical new backend, old ASR drafts unchanged.')
    tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
    bank={};qs=rows(HERE/'train_queries.jsonl');dq=rows(HERE/'dev_queries.jsonl')
    assert len(qs)==5325 and len(dq)==190
    for q in qs+dq:
        assert set(q)=={'id','edit_id','output_span','draft','input_evidence','A','B','evidence'}
        assert set(q['input_evidence'])=={'asr_top1','nbest','hotwords'}
    def encode(q,mode,task):
        key=(q['edit_id'],q['A']['role'],mode,task)
        assert key not in bank
        prompt=tok.apply_chat_template(render(q,mode,task),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(prompt,add_special_tokens=False,return_tensors='pt').input_ids
        assert ids.shape[0]==1 and 0<ids.shape[1]<=c['max_sequence']
        bank[key]=ids
    for i,q in enumerate(qs):
        qq=swap(q) if order_bit(q['edit_id']) else q
        for mode in ('none','true'):
            for task in c['tasks']:encode(qq,mode,task)
        # Smoke eval uses original order, not necessarily training display order.
        if q['edit_id'] in old_d['smoke_ids'] and qq['A']['role']!=q['A']['role']:
            for mode in ('none','true'):encode(q,mode,'full')
        if (i+1)%1000==0:status('PRETOKENIZING_TRAIN',done=i+1,total=len(qs))
    for q in dq:
        for mode in c['evaluation_modes']:encode(q,mode,'full')
        encode(swap(q),'true','full')
    torch.save(bank,HERE/'token_bank.pt')
    save(HERE/'token_bank_metadata.json',dict(reference_used=False,entries=len(bank),max_tokens=max(t.shape[1] for t in bank.values()),
        token_bank_sha256=sha(HERE/'token_bank.pt'),train_queries_sha256=sha(HERE/'train_queries.jsonl'),dev_queries_sha256=sha(HERE/'dev_queries.jsonl')))
    save(HERE/'protocol.json',c)
    d=copy.deepcopy(old_d);d['protocol_sha256']=sha(HERE/'protocol.json')
    d['hashes']={str(HERE/x):sha(HERE/x) for x in files+['token_bank.pt','token_bank_metadata.json']}
    d['parent_data_lock_sha256']=sha(parent_dir/'data_lock.json')
    save(HERE/'data_lock.json',d)
    save(HERE/'reuse_audit.json',dict(status='PASS',parent_protocol_sha256=parent_hash,unchanged_candidate_order_labels=True,
        reused_files={x:sha(HERE/x) for x in files},pairs=5325,updates=666,official_initialization=c['checkpoint'],
        no_old_smoke_weights_reused=True,source_and_acceleration_hashes_locked=True,token_bank_reference_used=False))
    status('REUSE_PREPARED',pairs=5325,updates=666,token_bank_entries=len(bank))


def labels():
    from transformers import AutoTokenizer
    c=configuration();s=load(HERE/'train_evidence_seal.json')
    assert s['protocol_sha256']==c['protocol_sha256'] and not s['reference_used']
    assert s['evidence_sha256']==sha(HERE/'train_evidence.jsonl') and s['train_drafts_sha256']==sha(HERE/'train_drafts.jsonl')
    tr={x['id']:x for x in rows(c['train'])};qs=[q for x in rows(HERE/'train_evidence.jsonl') for q in x['queries']]
    assert qs and len({q['edit_id'] for q in qs})==len(qs)
    lab=[preference(q,tr[q['id']]['reference'],e.norm,lambda a,b:e.align(a,b)[0],'train') for q in qs]
    write_rows(HERE/'train_queries.jsonl',qs);write_rows(HERE/'train_labels.jsonl',lab)
    tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
    choice=[tok(k,add_special_tokens=False)['input_ids'] for k in ('A','B')]
    assert all(len(x)==1 for x in choice) and choice[0]!=choice[1]
    lengths=[]
    for q in qs:
        for mode in ('none','true'):
            for task in c['tasks']:
                for qq in (q,swap(q)):
                    text=tok.apply_chat_template(render(qq,mode,task),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    n=len(tok(text,add_special_tokens=False)['input_ids']);assert n<=c['max_sequence'],q['edit_id'];lengths.append((n,q['edit_id']))
    dq=rows(HERE/'dev_queries.jsonl')
    for q in dq:
        for mode in c['evaluation_modes']:
            for qq in (q,swap(q)):
                n=len(tok(tok.apply_chat_template(render(qq,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)['input_ids'])
                assert n<=c['max_sequence']
    ids=[q['edit_id'] for q in qs];random.Random(917).shuffle(ids)
    smoke_ids=list(dict.fromkeys([qs[0]['edit_id'],next((q['edit_id'] for q,l in zip(qs,lab) if not l['neutral']),qs[0]['edit_id']),next((q['edit_id'] for q,l in zip(qs,lab) if l['neutral']),qs[0]['edit_id']),max(lengths)[1]]))
    hashes={str(HERE/x):sha(HERE/x) for x in ('train_drafts.jsonl','train_edit_manifest.jsonl','train_evidence.jsonl','train_evidence_seal.json','train_queries.jsonl','train_labels.jsonl')}
    lock=dict(protocol_sha256=c['protocol_sha256'],hashes=hashes,pairs=len(qs),updates=math.ceil(len(qs)/8),choice_token_ids=[x[0] for x in choice],train_order=ids,smoke_ids=smoke_ids,
        train_utterances=17294,utterances_with_pairs=len({q['id'] for q in qs}),neutral_pairs=sum(l['neutral'] for l in lab),prefer_A=sum(l['prob_A']==1 for l in lab),prefer_B=sum(l['prob_A']==0 for l in lab),max_prompt_tokens=max(lengths)[0],draft_failures=sum(not b['parsed'] for b in rows(HERE/'train_drafts.jsonl')))
    save(HERE/'data_lock.json',lock);status('TRAIN_DATA_LOCKED',pairs=lock['pairs'],updates=lock['updates'],neutral_pairs=lock['neutral_pairs'],max_prompt_tokens=lock['max_prompt_tokens'])


class Runtime:
    def __init__(self,c,d,training=False,checkpoint=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        assert c['numerical_settings']['deterministic_algorithms']
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        set_seed(917);torch.set_num_threads(6)
        self.c,self.d=c,d;self.tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
        cp=Path(checkpoint or c['checkpoint'])
        self.model=AutoModelForCausalLM.from_pretrained(c['base'],trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'})
        self.model=PeftModel.from_pretrained(self.model,cp/'adapter',is_trainable=training)
        saved=load_file(str(cp/'adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(self.model)
        assert saved.keys()==actual.keys() and all(torch.equal(v,actual[k].cpu()) for k,v in saved.items()),'Adapter reload identity'
        self.params=[v for v in self.model.parameters() if v.requires_grad]
        self.initial_hash=sha(cp/'adapter/adapter_model.safetensors')
        if training:
            assert self.params and all('lora_' in n for n,v in self.model.named_parameters() if v.requires_grad)
            if c['acceleration']['gradient_checkpointing']:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:self.model.gradient_checkpointing_disable()
            self.model.enable_input_require_grads();self.model.train()
        else:
            self.model.eval()
            assert not self.params
        self.model.config.use_cache=False
        backends=[m.chunk_gated_delta_rule.__module__ for m in self.model.modules() if m.__class__.__name__=='Qwen3_5GatedDeltaNet']
        assert backends and all(x.startswith('fla.') for x in backends),backends
        self.bank=torch.load(HERE/'token_bank.pt',map_location='cpu',weights_only=True);self.gpu_tokens={}
    def logits(self,q,mode,task='full'):
        import torch
        key=(q['edit_id'],q['A']['role'],mode,task)
        if key not in self.gpu_tokens:
            ids=self.bank[key].cuda()
            self.gpu_tokens[key]=(ids,torch.ones_like(ids),torch.arange(ids.shape[1],device='cuda')[None])
        ids,mask,pos=self.gpu_tokens[key]
        assert 0<ids.shape[1]<=self.c['max_sequence'] and ids.shape[0]==1
        result=self.model(inputs_embeds=self.model.get_input_embeddings()(ids),attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=1)
        return result.logits[0,-1,self.d['choice_token_ids']].float()
    def optimizer(self):
        import torch
        from transformers import get_linear_schedule_with_warmup
        opt=torch.optim.AdamW(self.params,lr=self.c['lr'],weight_decay=self.c['weight_decay'])
        sch=get_linear_schedule_with_warmup(opt,min(self.c['warmup'],self.d['updates']),self.d['updates'])
        return opt,sch


def kernel_preflight():
    """Train-only Nbest smoke before expensive drafts; never formal initializer."""
    import torch
    from transformers import AutoTokenizer
    c=configuration();tr=rows(c['train']);found=[]
    # Candidate construction uses input Nbest only; no reference filtering.
    for row in tr:
        inp=r.whitelist(row['input'])
        for draft in inp['nbest']:
            manifest=e.candidates(row['id'],inp['asr_top1'],draft)
            edits=[z for z in manifest['edits'] if z['kind']=='different_phone_substitution']
            if edits:
                scores={m:{'original':{'score':None},'modified':{'score':None},'modified_minus_original':None} for m in ('native','phonetic_marginal')}
                q=r.make_query(manifest,edits[0],row['input'],{'true':scores})
                label=preference(q,row['reference'],e.norm,lambda a,b:e.align(a,b)[0],'train')
                found.append((q,label));break
        if len(found)==4:break
    assert found,'No train-only preflight candidate'
    tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
    ids=[tok(x,add_special_tokens=False)['input_ids'] for x in ('A','B')];assert all(len(x)==1 for x in ids)
    # Warmup scheduler begins at lr0: need two steps to test an actual update.
    d={'choice_token_ids':[x[0] for x in ids],'updates':4}
    with w.gpu_guard(p):
        rt=Runtime(c,d,True);opt,sch=rt.optimizer();versions={n:v._version for n,v in rt.model.named_parameters() if not v.requires_grad}
        initial=[v.detach().cpu().clone() for v in rt.params]
        first=update(rt,found,opt,sch,'TEXT');audit=update(rt,found,opt,sch,'TEXT')
        assert any(not torch.equal(a,v.detach().cpu()) for a,v in zip(initial,rt.params)),'LoRA did not change after warmup step'
        assert all(v._version==versions[n] for n,v in rt.model.named_parameters() if not v.requires_grad)
        save(HERE/'kernel_preflight.json',dict(status='PASS',train_only_ids=[q['id'] for q,l in found],candidate_source='Input Nbest, not frozen drafts; kernel/memory smoke only, not formal training data',zero_lr_first_update=first,update=audit,lora_changed=True,frozen_weights_unchanged=True,lora_trainable_parameters=sum(v.numel() for v in rt.params),peak_memory_allocated=int(torch.cuda.max_memory_allocated()),choice_token_ids=d['choice_token_ids'],formal_training_started=False))
    status('KERNEL_PREFLIGHT_PASS',loss=audit['loss'])


def update(rt,chunk,opt,sch,arm):
    import torch
    opt.zero_grad(set_to_none=True);values=[]
    for q,label in chunk:
        reverse=bool(order_bit(q['edit_id']));qq=swap(q) if reverse else q;y=label_for(q,label,reverse)
        target=torch.tensor([y,1-y],device='cuda',dtype=torch.float32)
        for task,weight in [('full',.5),('local',.5)]:
            logits=rt.logits(qq,'true' if arm=='AUDIO' else 'none',task)
            loss=-(target*torch.log_softmax(logits,dim=0)).sum()
            (loss*weight/len(chunk)).backward();values.append(loss.detach()*weight)
    # Avoid a CUDA-to-CPU synchronization at every task. Still reject all
    # nonfinite losses before optimizer.step(), never alter mathematical loss.
    logged=torch.stack(values)
    assert torch.isfinite(logged).all()
    assert all(v.grad is None for v in rt.model.parameters() if not v.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(rt.params,rt.c['clip']);assert torch.isfinite(norm) and norm>0
    opt.step();sch.step()
    return dict(loss=sum(logged.cpu().tolist())/len(chunk),grad_norm=float(norm),lr=sch.get_last_lr()[0])


def checkpoint(rt,opt,sch,path,step,examples,arm):
    import numpy as np
    import torch
    path.mkdir(exist_ok=False);rt.model.save_pretrained(path/'adapter')
    torch.save(dict(optimizer=opt.state_dict(),scheduler=sch.state_dict(),step=step,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path/'resume.pt')
    save(path/'metadata.json',dict(complete=True,arm=arm,step=step,examples=examples,protocol_sha256=rt.c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json'),initial_adapter_hash=sha(Path(rt.c['checkpoint'])/'adapter/adapter_model.safetensors'),hashes={str(x.relative_to(path)):sha(x) for x in path.rglob('*') if x.is_file()}))


def verify_checkpoint(cp,c,arm):
    meta=load(cp/'metadata.json');assert meta['complete'] and meta['arm']==arm and meta['protocol_sha256']==c['protocol_sha256'] and meta['data_lock_sha256']==sha(HERE/'data_lock.json')
    for name,digest in meta['hashes'].items():assert sha(cp/name)==digest,name
    return meta


def smoke(arm,reload=False):
    import numpy as np
    import torch
    c,d=data_config();qs={q['edit_id']:q for q in rows(HERE/'train_queries.jsonl')};ls={l['edit_id']:l for l in rows(HERE/'train_labels.jsonl')}
    chunk=[(qs[i],ls[i]) for i in d['smoke_ids']];directory=HERE/arm;directory.mkdir(exist_ok=True);cp=directory/'smoke_checkpoint'
    with w.gpu_guard(p):
        rt=Runtime(c,d,True,cp if reload else None);opt,sch=rt.optimizer()
        if reload:
            verify_checkpoint(cp,c,arm);state=torch.load(cp/'resume.pt',map_location='cpu',weights_only=False)
            opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            audit=load(directory/'smoke.json');rt.model.eval()
            with torch.no_grad():actual=rt.logits(chunk[0][0],'true' if arm=='AUDIO' else 'none').cpu().tolist()
            assert max(abs(a-b) for a,b in zip(actual,audit['reload_logits']))<=.005,'Reload logits changed'
            rt.model.train();res=update(rt,chunk,opt,sch,arm)
            assert abs(res['loss']-audit['resumed_update']['loss'])<=.005,'Resumed loss differs'
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            expected=load_file(str(directory/'smoke_after_resume/adapter_model.safetensors'));actual=get_peft_model_state_dict(rt.model)
            assert expected.keys()==actual.keys() and all(torch.equal(v,actual[k].cpu()) for k,v in expected.items()),'Resumed adapter differs'
            save(directory/'smoke_reload.json',dict(status='PASS',exact_resumed_adapter=True,reload_logits_tolerance=.005,resumed_update=res));status('SMOKE_RELOAD_PASS',arm=arm)
        else:
            versions={n:v._version for n,v in rt.model.named_parameters() if not v.requires_grad}
            first=update(rt,chunk,opt,sch,arm);res=update(rt,chunk,opt,sch,arm)
            assert all(v._version==versions[n] for n,v in rt.model.named_parameters() if not v.requires_grad),'Frozen weights changed'
            checkpoint(rt,opt,sch,cp,2,2*len(chunk),arm)
            assert sha(cp/'adapter/adapter_model.safetensors')!=rt.initial_hash,'LoRA unchanged'
            rt.model.eval()
            with torch.no_grad():ll=rt.logits(chunk[0][0],'true' if arm=='AUDIO' else 'none').cpu().tolist()
            rt.model.train();res2=update(rt,chunk,opt,sch,arm);rt.model.save_pretrained(directory/'smoke_after_resume')
            save(directory/'smoke.json',dict(status='PASS',initial_adapter_hash=rt.initial_hash,zero_lr_first_update=first,update=res,reload_logits=ll,resumed_update=res2,trainable_parameters=sum(v.numel() for v in rt.params),frozen_unchanged=True,peak_memory_allocated=int(torch.cuda.max_memory_allocated())))
            status('SMOKE_PASS',arm=arm,trainable_parameters=sum(v.numel() for v in rt.params))


def train(arm):
    import torch
    c,d=data_config();directory=HERE/arm;assert load(directory/'smoke_reload.json')['status']=='PASS'
    qs={q['edit_id']:q for q in rows(HERE/'train_queries.jsonl')};ls={l['edit_id']:l for l in rows(HERE/'train_labels.jsonl')}
    final=directory/'final_checkpoint';assert not final.exists()
    with w.gpu_guard(p):
        rt=Runtime(c,d,True);opt,sch=rt.optimizer();began=time.time()
        assert rt.initial_hash==load(directory/'smoke.json')['initial_adapter_hash']
        write_rows(directory/'train_order.jsonl',[{'edit_id':i} for i in d['train_order']])
        with (directory/'training.jsonl').open('x',encoding='utf-8') as f:
            for step,start in enumerate(range(0,d['pairs'],c['accumulation']),1):
                ii=d['train_order'][start:start+c['accumulation']];chunk=[(qs[i],ls[i]) for i in ii]
                rec=update(rt,chunk,opt,sch,arm);rec.update(step=step,pairs=start+len(ii),seconds=time.time()-began)
                f.write(json.dumps(rec)+'\n');f.flush()
                if step==1 or step%10==0 or step==d['updates']:
                    status('TRAINING',arm=arm,step=step,total_updates=d['updates'],pairs=rec['pairs'],loss=rec['loss'],seconds=rec['seconds'],remaining_seconds_estimate=rec['seconds']/step*(d['updates']-step))
                    save(directory/'status.json',dict(status='TRAINING',**rec,total_updates=d['updates']))
                if step%c['checkpoint_interval']==0:checkpoint(rt,opt,sch,directory/f'checkpoint_{step}',step,rec['pairs'],arm)
            checkpoint(rt,opt,sch,final,d['updates'],d['pairs'],arm)
        save(directory/'selected_checkpoint.json',dict(checkpoint=str(final),selection='Fixed final one-epoch checkpoint; no dev selection',updates=d['updates'],pairs=d['pairs']))
        save(directory/'status.json',dict(status='TRAINED_NOT_VALIDATED',updates=d['updates'],pairs=d['pairs']))
    status('TRAIN_COMPLETE',arm=arm,updates=d['updates'])


def evaluate(arm):
    import torch
    c,d=data_config();directory=HERE/arm;directory.mkdir(exist_ok=True)
    checkpoint_path=None if arm=='UNADAPTED' else directory/'final_checkpoint'
    if checkpoint_path is not None:verify_checkpoint(checkpoint_path,c,arm)
    qs=rows(HERE/'dev_queries.jsonl');assert len(qs)==190
    with w.gpu_guard(p):
        rt=Runtime(c,d,False,checkpoint_path);began=time.time();hashes={}
        for mode in c['evaluation_modes']+['true_reversed']:
            path=directory/(mode+'.decisions.jsonl');partial=Path(str(path)+'.partial');assert not path.exists() and not partial.exists()
            with partial.open('x',encoding='utf-8') as f,torch.inference_mode():
                for i,q in enumerate(qs):
                    qq=swap(q) if mode=='true_reversed' else q;condition='true' if mode=='true_reversed' else mode
                    ll=rt.logits(qq,condition).cpu().tolist();key=choose(*ll)
                    x=dict(id=q['id'],edit_id=q['edit_id'],arm=arm,mode=mode,logits=ll,selected_key=key,selected_role=qq[key]['role'],candidate_roles={k:qq[k]['role'] for k in ('A','B')},batch_size=1,padding_tokens=0)
                    f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
                    if (i+1)%50==0 or i+1==190:status('DECISION_VALIDATION',arm=arm,mode=mode,done=i+1,total=190,seconds=time.time()-began)
            assert [x['edit_id'] for x in rows(partial)]==[q['edit_id'] for q in qs];partial.rename(path);hashes[mode]=sha(path)
        save(directory/'decision_seal.json',dict(protocol_sha256=c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json'),hashes=hashes,reference_used=False,checkpoint=str(checkpoint_path or c['checkpoint'])))
    status('DECISIONS_SEALED',arm=arm)


def report():
    c,d=data_config();old=rows(c['baseline']);qs=rows(HERE/'dev_queries.jsonl');baseline=p.metrics(old,protected=False)
    assert (baseline['summary']['n'],baseline['summary']['chars'],baseline['summary']['mentions'],baseline['summary']['errors'],baseline['summary']['hits'])==(1334,21102,599,622,469)
    # Reference diagnosis AFTER all three arms are sealed. Never used for training/selection.
    labels_by={z['edit_id']:z for z in load(ROOT/'acoustic_review_20261005/posthoc_cases.json')}
    metrics={};cells={};order_agreement={};hashes={}
    for arm in ('UNADAPTED','AUDIO','TEXT'):
        directory=HERE/arm;seal=load(directory/'decision_seal.json')
        assert seal['protocol_sha256']==c['protocol_sha256'] and not seal['reference_used']
        for mode,digest in seal['hashes'].items():assert sha(directory/(mode+'.decisions.jsonl'))==digest
        normal=rows(directory/'true.decisions.jsonl');rev=rows(directory/'true_reversed.decisions.jsonl')
        order_agreement[arm]=sum(a['selected_role']==b['selected_role'] for a,b in zip(normal,rev))
        for mode in c['evaluation_modes']:
            ds=rows(directory/(mode+'.decisions.jsonl'));assert [x['edit_id'] for x in ds]==[q['edit_id'] for q in qs]
            selected={};cc={k:dict(n=0,original=0,modified=0) for k in ('harmful','helpful','neutral')}
            for q,x in zip(qs,ds):
                assert choose(*x['logits'])==x['selected_key'];selected.setdefault(q['id'],[]).append((q,x['selected_role']))
                z=cc[labels_by[q['edit_id']]['diagnosis']];z['n']+=1;z[x['selected_role']]+=1
            out=[]
            for b in old:
                text=e.norm(b['prediction']) if b['id'] in selected else b['prediction']
                for q,role in reversed(selected.get(b['id'],[])):
                    lo,hi=q['output_span'];orig=next(q[k] for k in ('A','B') if q[k]['role']=='original');mod=next(q[k] for k in ('A','B') if q[k]['role']=='modified')
                    assert text[lo:hi]==mod['text']
                    if role=='original':text=text[:lo]+orig['text']+text[hi:]
                out.append(dict(id=b['id'],input=b['input'],reference=b['reference'],prediction=text,parsed=True,condition=arm+'__'+mode))
            assert len(out)==1334 and [x['id'] for x in out]==[x['id'] for x in old]
            write_rows(directory/(mode+'.predictions.jsonl'),out);mm=p.metrics(out,protected=False);mm['diagnostics']=r.diagnose(out,p.evaluator().norm)
            assert (mm['summary']['n'],mm['summary']['chars'],mm['summary']['mentions'])==(1334,21102,599)
            save(directory/(mode+'.metrics.json'),mm);key=arm+'__'+mode;metrics[key]=mm;cells[key]=cc;hashes[key]=sha(directory/(mode+'.predictions.jsonl'))
    contrasts={}
    for arm in ('AUDIO','TEXT'):
        a=arm+'__true'
        contrasts[a+'_vs_old']=p.bootstrap(metrics[a]['per_utterance'],baseline['per_utterance'])
        for mode in ('none','zero','mismatch'):
            b=arm+'__'+mode;contrasts[a+'_vs_'+b]=p.bootstrap(metrics[a]['per_utterance'],metrics[b]['per_utterance'])
    for b in ('TEXT__none','TEXT__true','UNADAPTED__true','UNADAPTED__none'):
        contrasts['AUDIO__true_vs_'+b]=p.bootstrap(metrics['AUDIO__true']['per_utterance'],metrics[b]['per_utterance'])
    save(HERE/'summary.json',dict(status='COMPLETE',protocol_sha256=c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json'),baseline=baseline['summary'],summaries={k:v['summary'] for k,v in metrics.items()},diagnostics={k:v['diagnostics'] for k,v in metrics.items()},decisions=cells,paired95ci=contrasts,true_order_agreement_of190=order_agreement,prediction_hashes=hashes,limitations=c['limitations']))
    status('ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def pipeline():
    jobs=[('prepare-reuse',None),('evaluate','UNADAPTED'),('smoke','AUDIO'),('smoke-reload','AUDIO'),('smoke','TEXT'),('smoke-reload','TEXT'),('train','AUDIO'),('evaluate','AUDIO'),('train','TEXT'),('evaluate','TEXT'),('report',None)]
    for cmd,arm in jobs:
        status('PIPELINE_STAGE_START',command=cmd,arm=arm)
        log=HERE/(cmd+('_'+arm if arm else '')+'.log')
        with log.open('x',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(HERE/'experiment.py'),cmd]+([arm] if arm else []),stdout=f,stderr=subprocess.STDOUT,check=True,cwd=HERE)
    status('ALL_DONE')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pipeline','prepare-reuse','smoke','smoke-reload','train','evaluate','report']);ap.add_argument('arm',nargs='?',choices=['AUDIO','TEXT','UNADAPTED']);a=ap.parse_args()
    try:
        if a.command=='prepare-reuse':prepare_reuse()
        elif a.command=='smoke-reload':smoke(a.arm,True)
        elif a.command in ('smoke','train','evaluate'):globals()[a.command](a.arm)
        else:globals()[a.command]()
    except BaseException:
        failure=dict(error=traceback.format_exc(),when=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=a.command,arm=a.arm)
        save(HERE/('failure_'+str(time.time_ns())+'.json'),failure);status('FAILED',**failure);raise
