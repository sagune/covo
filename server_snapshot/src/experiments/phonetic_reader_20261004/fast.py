"""Audited execution-only acceleration; original reader/protocol stay untouched."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
import reader as original

HERE=Path(__file__).resolve().parent
w=original.w
BASE_RUNTIME=original.Runtime
BASE_UPDATE=original.update
BASE_CHECKPOINT=original.checkpoint

def collate(batches):
    from torch.nn.utils.rnn import pad_sequence
    inputs=pad_sequence([b['inputs_embeds'][0] for b in batches],batch_first=True,padding_value=0.)
    labels=pad_sequence([b['labels'][0] for b in batches],batch_first=True,padding_value=-100)
    lengths=torch.tensor([b['inputs_embeds'].shape[1] for b in batches],device=inputs.device)
    mask=(torch.arange(inputs.shape[1],device=inputs.device)[None]<lengths[:,None]).long()
    return dict(inputs_embeds=inputs,attention_mask=mask),labels

def target_loss(model,inputs,labels):
    # Project only hidden positions that predict a supervised answer token.
    # Keep exact per-example mean token CE, not a token-weighted batch average.
    shifted=labels[:,1:]; selected=shifted.ne(-100).any(0).nonzero().flatten()
    assert len(selected)
    output=model(**inputs,logits_to_keep=selected,use_cache=False)
    target=shifted.index_select(1,selected)
    logits=output.logits.float()
    losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
    counts=target.ne(-100).sum(1); assert (counts>0).all()
    return losses.sum(1)/counts

class FastRuntime(BASE_RUNTIME):
    def __init__(self,*args,**kwargs):
        # Load the original model FIRST: preserve original RMSNorm/conv modules.
        super().__init__(*args,**kwargs)
        self._kernels=[]; self.kernel_enabled=False; self.microbatch=1
        path=HERE/'accelerator_packages'; sys.path.insert(0,str(path))
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        for module in self.model.modules():
            if isinstance(module,q.Qwen3_5GatedDeltaNet):
                self._kernels.append((module,module.chunk_gated_delta_rule,module.recurrent_gated_delta_rule,
                                     chunk_gated_delta_rule,fused_recurrent_gated_delta_rule))
        assert len(self._kernels)==24

    def activate(self,audio,reader=True):
        super().activate(audio,reader)
        for module,old_chunk,old_recurrent,fast_chunk,fast_recurrent in getattr(self,'_kernels',[]):
            # Text is always the historical fallback backend, even in eval.
            module.chunk_gated_delta_rule=fast_chunk if self.kernel_enabled and audio else old_chunk
            module.recurrent_gated_delta_rule=fast_recurrent if self.kernel_enabled and audio else old_recurrent

def losses_and_backward(rt,chunk,legacy=False):
    ce=[]; auxiliary=[]; denominator=len(chunk)
    if legacy:
        for row in chunk:
            value=rt.model(**rt.batch(row,training=True),use_cache=False).loss
            (value/denominator).backward(); ce.append(float(value.detach()))
            if row['id'] in rt.aux:
                value=rt.model(**rt.batch(row,training=True,auxiliary=True),use_cache=False).loss
                (rt.c['auxiliary_weight']*value/denominator).backward(); auxiliary.append(float(value.detach()))
    else:
        for is_aux in (False,True):
            eligible=[r for r in chunk if not is_aux or r['id'] in rt.aux]
            for start in range(0,len(eligible),rt.microbatch):
                rows=eligible[start:start+rt.microbatch]
                inputs,labels=collate([rt.batch(row,training=True,auxiliary=is_aux) for row in rows])
                values=target_loss(rt.model,inputs,labels); assert torch.isfinite(values).all()
                weight=rt.c['auxiliary_weight'] if is_aux else 1.
                (weight*values.sum()/denominator).backward()
                (auxiliary if is_aux else ce).extend(values.detach().cpu().tolist())
    assert len(ce)==len(chunk)
    return dict(correction_ce=sum(ce)/len(ce),local_phone_ce=sum(auxiliary)/len(auxiliary) if auxiliary else None,
                local_phone_n=len(auxiliary),weighted_objective=(sum(ce)+rt.c['auxiliary_weight']*sum(auxiliary))/denominator)

def update(rt,chunk,opt,sch):
    opt.zero_grad(set_to_none=True); result=losses_and_backward(rt,chunk)
    assert all(x.grad is None for n,x in rt.model.named_parameters() if '.evidence.' not in n)
    reader=torch.stack([x.grad.float().norm() for x in rt.evidence.values() if x.grad is not None]).sum()
    connector=torch.stack([x.grad.float().norm() for x in rt.connector.parameters() if x.grad is not None]).sum()
    assert torch.isfinite(reader+connector) and reader>0 and connector>0
    grad=torch.nn.utils.clip_grad_norm_(rt.parameters,1.); assert torch.isfinite(grad)
    opt.step(); sch.step()
    result.update(reader_grad=float(reader),connector_grad=float(connector),grad_norm=float(grad),
                  microbatch=rt.microbatch,fla_delta=rt.kernel_enabled)
    return result

def gradients(rt):
    groups=[list(rt.evidence.values()),[p for p in rt.connector.parameters() if p.requires_grad]]
    return [torch.cat([p.grad.detach().float().flatten().cpu() for p in group if p.grad is not None]) for group in groups]

def comparison(a,b):
    cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    relative=float((a-b).norm()/a.norm().clamp_min(1e-12))
    return dict(cosine=cosine,relative_l2=relative)

def manifest_sources():
    files=[HERE/name for name in ('fast.py','test_fast.py','OPTIMIZATION.md')]
    packages=HERE/'accelerator_packages'
    files += sorted(p for p in packages.rglob('*') if p.is_file() and (p.suffix in ('.py','.so') or p.name in ('METADATA','RECORD')))
    return {str(p.resolve()):w.sha(p) for p in files}

def benchmark():
    c,p=original.configuration(); d=HERE/'R'
    with w.gpu_guard(p):
        rt=FastRuntime(c,p,True,Path(d/'smoke_checkpoint'))
        rows=p.load(c['train']); by_id={r['id']:r for r in rows}
        chunk=[by_id[c['smoke_ids'][0]],by_id[c['smoke_ids'][1]],rows[1],rows[2]]
        initial={n:x._version for n,x in rt.model.named_parameters() if '.evidence.' not in n}
        rt.model.zero_grad(set_to_none=True); rt.connector.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); began=time.perf_counter(); ref=losses_and_backward(rt,chunk,legacy=True)
        torch.cuda.synchronize(); baseline_seconds=time.perf_counter()-began; reference_grad=gradients(rt)
        candidates=[]; chosen=None
        # Predetermined hardware choices, NOT selection on dev or model accuracy.
        for fla,micro in [(True,4),(True,2),(False,4),(False,2),(False,1)]:
            rt.kernel_enabled=fla; rt.microbatch=micro
            try:
                # Warm/compile before measuring stable throughput.
                rt.model.zero_grad(set_to_none=True); rt.connector.zero_grad(set_to_none=True)
                losses_and_backward(rt,chunk)
                rt.model.zero_grad(set_to_none=True); rt.connector.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); began=time.perf_counter()
                result=losses_and_backward(rt,chunk); torch.cuda.synchronize(); seconds=time.perf_counter()-began
                checks=[comparison(a,b) for a,b in zip(reference_grad,gradients(rt))]
                difference=abs(ref['weighted_objective']-result['weighted_objective'])
                peak=torch.cuda.max_memory_allocated()/1e9
                passed=(difference<=.01+.01*abs(ref['weighted_objective']) and
                        all(x['cosine']>=.99 and x['relative_l2']<=.15 for x in checks) and peak<=30.)
                rec=dict(fla=fla,microbatch=micro,seconds=seconds,reference_seconds=baseline_seconds,
                         speedup=baseline_seconds/seconds,loss_difference=difference,gradients=checks,peak_gpu_gb=peak,passed=passed)
                candidates.append(rec); print(json.dumps(rec),flush=True)
                if passed and seconds<baseline_seconds: chosen=rec; break
            except torch.OutOfMemoryError:
                rt.model.zero_grad(set_to_none=True); rt.connector.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                rec=dict(fla=fla,microbatch=micro,passed=False,reason='OOM'); candidates.append(rec); print(json.dumps(rec),flush=True)
        assert chosen, 'No numerically verified faster configuration; preserve existing runner'
        assert all(x._version==initial[n] for n,x in rt.model.named_parameters() if n in initial)
        # Worst-case memory probe: four copies of train-only longest bound example.
        rt.kernel_enabled=chosen['fla']; rt.microbatch=chosen['microbatch']
        rt.model.zero_grad(set_to_none=True); rt.connector.zero_grad(set_to_none=True)
        losses_and_backward(rt,[by_id[c['smoke_ids'][1]]]*4)
        assert torch.cuda.max_memory_allocated()/1e9<=30., 'Worst-case batch exceeds declared30GB budget'
        info=dict(status='BENCHMARK_PASS',protocol_sha256=c['protocol_sha256'],choice=chosen,candidates=candidates,
                  sources=manifest_sources(),torch=torch.__version__,triton=__import__('triton').__version__,
                  original_R_status=w.read(d/'status.json'),numerical='FP32 per-example CE; same targets/weights/order/effective8; batched/kernel roundoff expected')
        assert not (HERE/'optimization.json').exists()
        w.save(HERE/'optimization.json',info)
        # Independent fast optimizer save/reload smoke; formal init remains original.
        opt,sch=rt.optimizer(); meta=original.restore(rt,opt,sch,d/'smoke_checkpoint')
        rt.kernel_enabled=chosen['fla']; rt.microbatch=chosen['microbatch']
        update(rt,chunk,opt,sch)
        q=d/'fast_smoke_checkpoint'; BASE_CHECKPOINT(rt,opt,sch,q,meta['step']+1,meta['examples']+len(chunk))
        rt.model.eval(); rt.connector.eval()
        w.save(d/'fast_smoke_before_reload.json',dict(true=rt.generate(chunk[0],'true'),text=rt.generate(chunk[0],'text')))

def read_optimization():
    info=w.read(HERE/'optimization.json')
    for path,expected in info['sources'].items(): assert w.sha(path)==expected,('Optimized source changed',path)
    return info

class SelectedRuntime(FastRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); info=read_optimization()
        assert info['protocol_sha256']==self.c['protocol_sha256']
        self.kernel_enabled=info['choice']['fla']; self.microbatch=info['choice']['microbatch']

def reload_smoke():
    c,p=original.configuration(); d=HERE/'R'
    with w.gpu_guard(p):
        rt=SelectedRuntime(c,p,True); opt,sch=rt.optimizer()
        meta=original.restore(rt,opt,sch,d/'fast_smoke_checkpoint')
        state=torch.load(d/'fast_smoke_checkpoint/state.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(value,rt.evidence[name].cpu()) for name,value in state['evidence'].items())
        assert all(torch.equal(value,rt.connector.state_dict()[name].cpu()) for name,value in state['connector'].items())
        assert sch.last_epoch==meta['step'] and all(int(value['step'])==meta['step'] for value in opt.state.values())
        row={r['id']:r for r in p.load(c['train'])}[c['smoke_ids'][0]]
        rt.model.eval(); rt.connector.eval(); saved=w.read(d/'fast_smoke_before_reload.json')
        assert rt.generate(row,'true')['model_output']==saved['true']['model_output']
        assert rt.generate(row,'text')['model_output']==saved['text']['model_output']
        rt.model.train(); rt.connector.train(); record=update(rt,[row],opt,sch)
        w.save(d/'fast_smoke.json',dict(status='PASS',protocol_sha256=c['protocol_sha256'],optimization_sha256=w.sha(HERE/'optimization.json'),resumed_update=record))

def checkpoint(rt,opt,sch,path,step,examples):
    BASE_CHECKPOINT(rt,opt,sch,path,step,examples)
    meta=w.read(path/'metadata.json'); meta['optimization_sha256']=w.sha(HERE/'optimization.json'); w.save(path/'metadata.json',meta)

def dispatch(command):
    if command=='benchmark': benchmark(); return
    if command=='reload-fast-smoke': reload_smoke(); return
    info=read_optimization(); assert w.read(HERE/'R/fast_smoke.json')['optimization_sha256']==w.sha(HERE/'optimization.json')
    original.Runtime=SelectedRuntime; original.update=update; original.checkpoint=checkpoint
    if command=='train': original.train()
    elif command=='eval': original.evaluate()
    else:
        original.report(); result=w.read(HERE/'summary.json')
        result['optimization_sha256']=w.sha(HERE/'optimization.json'); result['execution_acceleration']=info['choice']; w.save(HERE/'summary.json',result)

def pipeline():
    import fcntl
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commands=['benchmark','reload-fast-smoke','train','eval','report']
        for command in commands:
            if command=='benchmark' and (HERE/'optimization.json').exists(): continue
            if command=='reload-fast-smoke' and (HERE/'R/fast_smoke.json').exists(): continue
            w.save(HERE/'status.json',dict(status='RUNNING_OPTIMIZED',phase=command,pid=os.getpid(),updated=time.time()))
            with (HERE/f'fast_{command}_{time.time_ns()}.log').open('x') as stream:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),command],stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['benchmark','reload-fast-smoke','train','eval','report','pipeline']); args=parser.parse_args()
    try:
        if args.command=='pipeline': pipeline()
        else: dispatch(args.command)
    except BaseException:
        record=dict(status='OPTIMIZATION_FAILED',command=args.command,error=traceback.format_exc(),time=time.time())
        w.save(HERE/f'optimization_failure_{time.time_ns()}.json',record); w.save(HERE/'status.json',record); raise
