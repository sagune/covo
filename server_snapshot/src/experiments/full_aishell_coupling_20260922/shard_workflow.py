"""Train/evaluate residual and MLP connectors on the existing CBSense shard."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import time
import traceback
import sys

HERE=Path(__file__).resolve().parent
RES=HERE.parent/'text_residual_coupling_20260920'
sys.path.insert(0,str(RES))
import workflow as w


def legacy():
    spec=importlib.util.spec_from_file_location('shard_legacy',HERE.parent/'staged_coupling_20260917'/'run.py')
    p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p); return p


def config(arch):
    d=HERE/'shard_data'; p=legacy(); base=w.read(RES/'runs'/('residual_v1' if arch=='residual' else 'mlp_B_control')/'protocol.json')
    tr=p.load(d/'train.jsonl'); dv=p.load(d/'dev.jsonl')
    c=dict(base); c.update(version=2,architecture=arch,legacy=str(HERE.parent/'staged_coupling_20260917'),train=str(d/'train.jsonl'),dev=str(d/'dev.jsonl'),train_n=len(tr),dev_n=len(dv),
      steps=math.ceil(len(tr)/8),accumulation=8,selection='Fixed final one-pass shard checkpoint',label='AISHELL CBSense shard diagnostic; not held-out',
      data_caveat=w.read(d/'protocol.json')['train_policy'],external_context_caveat=w.read(d/'protocol.json')['caveat'],
      mlp_initialization=str(HERE.parent/'staged_coupling_20260917'/'overnight'/'Afull_3243') if arch=='mlp' else None,
      audio_permutation=p.permutation(dv),modes=['text','true','zero','mismatch'])
    paths=[Path(x) for x in base['sources'] if Path(x).exists()]
    paths += [d/'protocol.json',d/'train.jsonl',d/'dev.jsonl',HERE/'shard_prepare.py',HERE/'shard_workflow.py']
    c['sources']={str(x.resolve()):w.sha(x) for x in paths}
    cache={}
    import numpy as np
    for r in tr+dv:
        q=p.RUN/'audio'/f"{r['id']}.npy"; x=np.load(q,mmap_mode='r',allow_pickle=False)
        assert x.ndim==2 and x.shape[1]==512 and len(x)
        cache[r['id']]=dict(path=str(q),sha256=w.sha(q),frames=len(x))
    dest=HERE/'runs'/('shard_'+arch); dest.mkdir(parents=True,exist_ok=False)
    w.save(dest/'cache_manifest.json',cache); c['cache_manifest_sha256']=w.sha(dest/'cache_manifest.json'); w.save(dest/'protocol.json',c)
    w.save(dest/'status.json',dict(status='PREPARED_SHARD',train_n=len(tr),dev_n=len(dv),training='NOT_STARTED'))
    return dest,c,p


def train(directory):
    import torch
    d=Path(directory); c,p=w.configuration(d); assert w.read(d/'gpu_smoke.json')['status']=='PASS'
    with w.gpu_guard(p):
        rt=w.Runtime(c,p,training=True); rows=p.load(c['train']); random.Random(c['seed']).shuffle(rows)
        opt=torch.optim.AdamW(rt.connector.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/c['warmup'] if s<c['warmup'] else max(0.,(c['steps']-s)/(c['steps']-c['warmup'])))
        started=time.time(); w.save(d/'training_started.json',dict(pid=__import__('os').getpid(),started=started))
        with (d/'training.jsonl').open('x') as log:
            for step in range(c['steps']):
                chunk=rows[step*c['accumulation']:(step+1)*c['accumulation']]; assert chunk
                opt.zero_grad(set_to_none=True); loss_sum=0.
                for row in chunk:
                    loss=rt.model(**rt.batch(row,training=True),use_cache=False).loss; assert torch.isfinite(loss)
                    (loss/len(chunk)).backward(); loss_sum+=float(loss.detach())
                grad=torch.nn.utils.clip_grad_norm_(rt.connector.parameters(),1.); assert torch.isfinite(grad) and grad>0
                opt.step(); sch.step(); rec=dict(status='TRAINING_SHARD',step=step+1,steps=c['steps'],examples=min((step+1)*8,len(rows)),loss=loss_sum/len(chunk),grad_norm=float(grad),seconds=time.time()-started)
                log.write(json.dumps(rec)+'\n'); log.flush(); w.save(d/'status.json',rec)
                if step==0 or (step+1)%10==0: print(json.dumps(rec),flush=True)
        final=d/'final'; final.mkdir(); torch.save(rt.connector.state_dict(),final/'connector.pt')
        w.save(final/'metadata.json',dict(architecture=c['architecture'],step=c['steps'],examples=len(rows),protocol_sha256=c['protocol_sha256'],connector_sha256=w.sha(final/'connector.pt'),complete=True))
        w.save(d/'selected_checkpoint.json',dict(checkpoint=str(final),connector_sha256=w.sha(final/'connector.pt'),reason=c['selection']))
        w.save(d/'status.json',dict(status='TRAINED_SHARD_NOT_EVALUATED'))


def evaluate(directory):
    import torch
    from diagnostics import diagnose
    d=Path(directory); c,p=w.configuration(d); rows=p.load(c['dev']); sel=w.read(d/'selected_checkpoint.json'); assert sel['connector_sha256']==w.sha(d/'final/connector.pt')
    with w.gpu_guard(p):
        rt=w.Runtime(c,p,checkpoint=d/'final'); out=d/'dev'; out.mkdir()
        mets={}
        for mode in c['modes']:
            path=out/f'{mode}.predictions.jsonl'; partial=Path(str(path)+'.partial')
            with partial.open('x') as f:
                for i,row in enumerate(rows):
                    pr=rt.predict(row,mode,c['audio_permutation'][row['id']] if mode=='mismatch' else None); f.write(json.dumps(pr,ensure_ascii=False)+'\n'); f.flush()
                    if (i+1)%50==0: print(json.dumps(dict(mode=mode,done=i+1,total=len(rows))),flush=True)
            partial.rename(path); pred=p.load(path); m=p.metrics(pred,protected=False); m['error_diagnostics']=diagnose(pred,p.evaluator().norm); mets[mode]=m; w.save(out/f'{mode}.metrics.json',m)
        text=p.load(out/'text.predictions.jsonl'); off=p.load(out/'off.predictions.jsonl'); assert [x['model_output'] for x in text]==[x['model_output'] for x in off]
        summary=dict(label=c['label'],summaries={k:v['summary'] for k,v in mets.items()},error_diagnostics={k:v['error_diagnostics'] for k,v in mets.items()},paired_bootstrap={m:p.bootstrap(mets['true']['per_utterance'],mets[m]['per_utterance']) for m in ('text','zero','mismatch')},no_output_repairs=True,normalization=c['normalization'],caveat=c['data_caveat'])
        w.save(out/'summary.json',summary); w.save(d/'status.json',dict(status='COMPLETED_SHARD_DEV',summary=str(out/'summary.json'))); print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    pa=argparse.ArgumentParser(); pa.add_argument('command',choices=['prepare','smoke','train','eval']); pa.add_argument('arch',choices=['residual','mlp']); a=pa.parse_args(); d=HERE/'runs'/('shard_'+a.arch)
    try:
        if a.command=='prepare': config(a.arch)
        elif a.command=='smoke': w.smoke(d)
        elif a.command=='train': train(d)
        else: evaluate(d)
    except BaseException:
        w.save(d/f'failure_{time.time_ns()}.json',dict(error=traceback.format_exc(),command=a.command)); raise

if __name__=='__main__': main()
