"""Memory-safe, resumable rerun of immutable all-edit contextual LoRA."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

# Must be set before importing any torch-dependent repository modules.
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/src/experiments')
FAILED=ROOT/'all_edit_review_20261007'
spec=importlib.util.spec_from_file_location('immutable_all_edit_v1',FAILED/'experiment.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.HERE=HERE;a.b.HERE=HERE
b=a.b
e,r,w,p=a.e,a.r,a.w,a.p
load,rows,sha,save,write_rows,status=a.load,a.rows,a.sha,a.save,a.write_rows,a.status
OriginalRuntime=b.Runtime
original_checkpoint=b.checkpoint


class SafeRuntime(OriginalRuntime):
    def logits(self,q,mode,task='full'):
        import torch
        assert self.gpu_tokens=={},'GPU token cache must remain empty'
        key=(q['edit_id'],q['A']['role'],mode,task)
        ids=self.bank[key].to('cuda:0',non_blocking=False)
        assert ids.shape[0]==1 and 0<ids.shape[1]<=self.c['max_sequence']
        mask=torch.ones_like(ids);pos=torch.arange(ids.shape[1],device='cuda:0')[None]
        result=self.model(inputs_embeds=self.model.get_input_embeddings()(ids),attention_mask=mask,
                          position_ids=pos,use_cache=False,logits_to_keep=1)
        return result.logits[0,-1,self.d['choice_token_ids']].float()


b.Runtime=SafeRuntime


def atomic_checkpoint(rt,opt,sch,path,step,examples,arm):
    path=Path(path);partial=Path(str(path)+'.partial')
    assert not path.exists() and not partial.exists(),'Preserve prior checkpoint artifacts'
    assert shutil.disk_usage(HERE).free>2*1024**3,'Checkpoint disk margin under2GiB'
    original_checkpoint(rt,opt,sch,partial,step,examples,arm)
    b.verify_checkpoint(partial,rt.c,arm)
    # POSIX rename makes only fully hash-checked checkpoints visible for resume.
    partial.rename(path)


b.checkpoint=atomic_checkpoint


def prepare_reuse():
    assert not (HERE/'protocol.json').exists()
    c0,d0=load(FAILED/'protocol.json'),load(FAILED/'data_lock.json')
    assert d0['protocol_sha256']==sha(FAILED/'protocol.json')
    for path,digest in {**c0['sources'],**d0['hashes']}.items():assert sha(path)==digest,path
    assert load(FAILED/'preparation_independent_audit.json')['status']=='PASS'
    assert (d0['pairs'],d0['updates'],d0['draft_failures'])==(18956,2370,49)
    assert not (FAILED/'AUDIO/final_checkpoint').exists() and not (FAILED/'TEXT/training.jsonl').exists()
    files=['train_drafts.jsonl','train_edit_manifest.jsonl','train_evidence.jsonl','train_queries.jsonl','train_labels.jsonl',
           'dev_queries.jsonl','token_bank.pt','token_bank_metadata.json','native_interface.json']
    for name in files:
        (HERE/name).symlink_to(FAILED/name);assert sha(HERE/name)==sha(FAILED/name)
    sources=dict(c0['sources'])
    dependencies=[FAILED/'experiment.py',FAILED/'protocol.json',FAILED/'data_lock.json',FAILED/'preparation_independent_audit.json']
    dependencies += [HERE/n for n in ['experiment.py','interface.py','test_interface.py','audit_prepared.py','audit_results.py','metric_auditor.py','PLAN.md','test_resume.py']]
    for path in dependencies:sources[str(path)]=sha(path)
    c=copy.deepcopy(c0)
    c.update(experiment='All-edit contextual review memory-safe resumable rerun',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             sources=sources,revision='v2: memory-only runtime revision after v1OOM; same data/prompts/labels/order/loss/hyperparameters',
             failed_parent_protocol=str(FAILED/'protocol.json'),checkpoint_interval=128,
             memory_policy=dict(allocator='expandable_segments:True',gpu_token_cache=False,activation_checkpointing='nonreentrant',
                                minimum_stress_headroom_bytes=4*1024**3,stress_updates_per_arm=16),
             recovery_policy='At most one fresh-process CUDA OOM retry per train arm, from latest complete hash-verified checkpoint; same locked configuration, no dropped samples',
             checkpoint_policy='Atomic complete model/optimizer/scheduler/Python+NumPy+CPU+CUDA RNG checkpoint every128 updates; fixed final2370 selected; no dev selection')
    c['acceleration']['gradient_checkpointing']=True
    c['limitations'] += ['Activation recomputation can change floating-point trajectory vs failedv1; both newarms use identical newbackend',
                         'Stress establishes headroom on this GPU/data, not a mathematical guarantee against external GPU use or hardware faults']
    save(HERE/'protocol.json',c);ph=sha(HERE/'protocol.json')
    for name in ('candidate_route_seal.json','train_evidence_seal.json'):
        obj=load(FAILED/name);obj['protocol_sha256']=ph;obj['reused_parent_seal_sha256']=sha(FAILED/name);save(HERE/name,obj)
    d=copy.deepcopy(d0);d['protocol_sha256']=ph
    d['hashes']={str(HERE/Path(path).name):sha(HERE/Path(path).name) for path in d0['hashes']}
    save(HERE/'data_lock.json',d)
    unchanged=['checkpoint','base','train','dev','baseline','train_n','dev_n','dev_queries','normalization','tasks','full_weight','local_weight',
               'expected_train_pairs','epochs','lr','weight_decay','accumulation','warmup','clip','max_sequence','seed','choice_contract','routing',
               'evaluation_modes','final_only','numerical_settings','train_type_counts','training_controls']
    for name in unchanged:assert c[name]==c0[name],name
    assert d['train_order']==d0['train_order'] and d['smoke_ids']==d0['smoke_ids']
    save(HERE/'reuse_audit.json',dict(status='PASS',failed_parent_protocol_sha256=sha(FAILED/'protocol.json'),
         same_model_initialization_data_labels_prompts_order_objective_hyperparameters=True,reused_files={n:sha(HERE/n) for n in files},
         only_runtime_checkpoint_storage_revision=True,no_failed928_step_weights_available=True,no_smoke_weights_as_formal_initializer=True))
    status('PREPARED_REUSED',pairs=18956,updates=2370)


def latest_checkpoint(directory,c,arm):
    candidates=[]
    for cp in directory.glob('checkpoint_*'):
        if cp.is_dir() and cp.name.split('_')[-1].isdigit() and (cp/'metadata.json').exists():
            meta=b.verify_checkpoint(cp,c,arm);candidates.append((meta['step'],cp))
    return max(candidates,default=(0,None),key=lambda x:x[0])[1]


def restore_training(rt,opt,sch,cp,c,arm):
    import numpy as np
    import torch
    meta=b.verify_checkpoint(cp,c,arm)
    assert meta['initial_adapter_hash']==sha(Path(c['checkpoint'])/'adapter/adapter_model.safetensors')
    state=torch.load(cp/'resume.pt',map_location='cpu',weights_only=False)
    assert state['step']==meta['step'] and meta['examples']==min(meta['step']*c['accumulation'],rt.d['pairs'])
    opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return meta['step']


def memory_stress(arm):
    import torch
    c,d=b.data_config();qs={q['edit_id']:q for q in rows(HERE/'train_queries.jsonl')};ls={x['edit_id']:x for x in rows(HERE/'train_labels.jsonl')}
    from interface import swap,order_bit
    mode='true' if arm=='AUDIO' else 'none'
    ordered=list(qs)
    # Read once. Longest chosen solely by token length, not label or dev result.
    bank=torch.load(HERE/'token_bank.pt',map_location='cpu',weights_only=True)
    lengths={uid:int(bank[(uid,(swap(qs[uid]) if order_bit(uid) else qs[uid])['A']['role'],mode,'full')].shape[1]) for uid in ordered}
    long_uid=max(ordered,key=lambda uid:lengths[uid]);ascending=sorted(ordered,key=lambda uid:(lengths[uid],uid))
    varying=[ascending[int(i*(len(ascending)-1)/7)] for i in range(8)]
    historical_failed=d['train_order'][928*8:929*8]
    with w.gpu_guard(p):
        rt=SafeRuntime(c,d,True);opt,sch=rt.optimizer();began=time.time();records=[]
        versions={n:v._version for n,v in rt.model.named_parameters() if not v.requires_grad}
        torch.cuda.reset_peak_memory_stats()
        for i in range(c['memory_policy']['stress_updates_per_arm']):
            ids=[long_uid]*8 if i%2==0 else (historical_failed if i%4==1 else varying)
            rec=a.update(rt,[(qs[u],ls[u]) for u in ids],opt,sch,arm);torch.cuda.synchronize()
            free,total=torch.cuda.mem_get_info();rec.update(update=i+1,token_lengths=[lengths[u] for u in ids],
                 allocated=int(torch.cuda.memory_allocated()),reserved=int(torch.cuda.memory_reserved()),free=int(free))
            assert rt.gpu_tokens=={}
            records.append(rec);status('MEMORY_STRESS',arm=arm,update=i+1,total=16,free_GiB=free/1024**3,seconds=time.time()-began)
        assert all(v._version==versions[n] for n,v in rt.model.named_parameters() if not v.requires_grad)
        peak_a=int(torch.cuda.max_memory_allocated());peak_r=int(torch.cuda.max_memory_reserved())
        free,total=torch.cuda.mem_get_info();external=total-free-int(torch.cuda.memory_reserved())
        headroom=total-max(external,0)-peak_r
        assert headroom>=c['memory_policy']['minimum_stress_headroom_bytes'],('Insufficient worst-case margin',headroom)
        save(HERE/('memory_stress_'+arm+'.json'),dict(status='PASS',arm=arm,train_only=True,discarded_weights=True,
             longest_id=long_uid,longest_tokens=lengths[long_uid],same_failed_batch_included=True,updates=16,
             peak_allocated=peak_a,peak_reserved=peak_r,total_memory=total,conservative_headroom_bytes=headroom,
             no_gpu_token_cache=True,frozen_weights_unchanged=True,seconds=time.time()-began,records=records,
             protocol_sha256=c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json')))
    status('MEMORY_STRESS_PASS',arm=arm,headroom_GiB=headroom/1024**3)


def train(arm):
    import torch
    c,d=b.data_config();directory=HERE/arm
    assert load(directory/'smoke_reload.json')['exact_resumed_adapter']
    assert load(HERE/('memory_stress_'+arm+'.json'))['status']=='PASS'
    qs={q['edit_id']:q for q in rows(HERE/'train_queries.jsonl')};ls={l['edit_id']:l for l in rows(HERE/'train_labels.jsonl')}
    final=directory/'final_checkpoint';assert not final.exists()
    prior=latest_checkpoint(directory,c,arm)
    attempt=len(list(directory.glob('training_attempt_*.jsonl')))+1
    orderfile=directory/'train_order.jsonl'
    if not orderfile.exists():write_rows(orderfile,[{'edit_id':uid} for uid in d['train_order']])
    assert [x['edit_id'] for x in rows(orderfile)]==d['train_order']
    with w.gpu_guard(p):
        rt=SafeRuntime(c,d,True,prior);opt,sch=rt.optimizer();start_step=restore_training(rt,opt,sch,prior,c,arm) if prior else 0
        assert start_step<d['updates'];began=time.time()
        log=directory/('training_attempt_'+str(attempt).zfill(3)+'.jsonl')
        save(directory/('attempt_'+str(attempt).zfill(3)+'.json'),dict(attempt=attempt,starting_completed_step=start_step,
             checkpoint=str(prior) if prior else c['checkpoint'],formal_initializer=c['checkpoint'],protocol_sha256=c['protocol_sha256'],
             data_lock_sha256=sha(HERE/'data_lock.json'),started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        with log.open('x',encoding='utf-8') as f:
            for step in range(start_step+1,d['updates']+1):
                lo=(step-1)*c['accumulation'];ids=d['train_order'][lo:lo+c['accumulation']]
                rec=a.update(rt,[(qs[u],ls[u]) for u in ids],opt,sch,arm)
                rec.update(step=step,pairs=lo+len(ids),attempt=attempt,seconds=time.time()-began,
                           allocated=int(torch.cuda.memory_allocated()),reserved=int(torch.cuda.memory_reserved()))
                f.write(json.dumps(rec)+'\n');f.flush()
                if step==start_step+1 or step%10==0 or step==d['updates']:
                    status('TRAINING',arm=arm,step=step,total_updates=d['updates'],pairs=rec['pairs'],loss=rec['loss'],
                           seconds=rec['seconds'],remaining_seconds_estimate=rec['seconds']/(step-start_step)*(d['updates']-step))
                    save(directory/'status.json',dict(status='TRAINING',**rec,total_updates=d['updates'],resumed_from_step=start_step))
                if step%c['checkpoint_interval']==0:
                    atomic_checkpoint(rt,opt,sch,directory/('checkpoint_'+str(step)),step,rec['pairs'],arm)
            atomic_checkpoint(rt,opt,sch,final,d['updates'],d['pairs'],arm)
        # Merge only records after each attempt's resume cursor, retaining all raw attempts.
        effective=[]
        for path in sorted(directory.glob('training_attempt_*.jsonl')):
            meta=load(directory/('attempt_'+path.stem.split('_')[-1]+'.json'))
            effective=[x for x in effective if x['step']<=meta['starting_completed_step']]+rows(path)
        assert [x['step'] for x in effective]==list(range(1,d['updates']+1))
        write_rows(directory/'training.jsonl',effective)
        save(directory/'selected_checkpoint.json',dict(checkpoint=str(final),selection='Fixed final2370 one-epoch checkpoint; no dev selection',updates=d['updates'],pairs=d['pairs']))
        save(directory/'status.json',dict(status='TRAINED_NOT_VALIDATED',updates=d['updates'],pairs=d['pairs'],attempts=attempt))
    status('TRAIN_COMPLETE',arm=arm,updates=d['updates'])


def pipeline():
    jobs=[('prepare-reuse',None),('audit-prepared',None),('smoke','AUDIO'),('smoke-reload','AUDIO'),('smoke','TEXT'),('smoke-reload','TEXT'),
          ('memory-stress','AUDIO'),('memory-stress','TEXT'),('train','AUDIO'),('train','TEXT'),('evaluate','UNADAPTED'),('evaluate','AUDIO'),
          ('evaluate','TEXT'),('report',None),('audit-results',None)]
    for cmd,arm in jobs:
        for attempt in range(1,3 if cmd=='train' else 2):
            status('PIPELINE_STAGE_START',command=cmd,arm=arm,attempt=attempt)
            path=HERE/(cmd+('_'+arm if arm else '')+'_attempt'+str(attempt)+'.log')
            with path.open('x',encoding='utf-8') as f:
                result=subprocess.run([sys.executable,'-u',str(HERE/'experiment.py'),cmd]+([arm] if arm else []),stdout=f,stderr=subprocess.STDOUT,cwd=HERE)
            if result.returncode==0:break
            # One immutable fresh-process retry only for explicit CUDA OOM with a complete resume point.
            oom='torch.OutOfMemoryError: CUDA out of memory' in path.read_text(encoding='utf-8')
            if cmd=='train' and attempt==1 and oom and latest_checkpoint(HERE/arm,b.configuration(),arm) is not None:
                status('SAME_PROTOCOL_OOM_RESUME',arm=arm);continue
            raise RuntimeError('Stage failed; preserved '+str(path))
    status('ALL_DONE_AUDITED')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pipeline','prepare-reuse','audit-prepared','smoke','smoke-reload','memory-stress','train','evaluate','report','audit-results'])
    ap.add_argument('arm',nargs='?',choices=['AUDIO','TEXT','UNADAPTED']);args=ap.parse_args()
    try:
        if args.command=='prepare-reuse':prepare_reuse()
        elif args.command=='smoke-reload':b.smoke(args.arm,True)
        elif args.command=='smoke':b.smoke(args.arm)
        elif args.command=='memory-stress':memory_stress(args.arm)
        elif args.command=='train':train(args.arm)
        elif args.command=='evaluate':a.evaluate(args.arm)
        elif args.command=='report':a.report()
        elif args.command in ('audit-prepared','audit-results'):
            subprocess.run([sys.executable,str(HERE/('audit_prepared.py' if args.command=='audit-prepared' else 'audit_results.py')),str(HERE)],check=True)
        else:pipeline()
    except BaseException:
        failure=dict(error=traceback.format_exc(),when=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=args.command,arm=args.arm)
        save(HERE/('failure_'+str(time.time_ns())+'.json'),failure);status('FAILED',**failure);raise
