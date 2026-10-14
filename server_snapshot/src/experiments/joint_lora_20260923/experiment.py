"""Locked AISHELL adapter co-adaptation diagnostic; no implicit full-data jobs."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'full_aishell_coupling_20260922'
RES = HERE.parent / 'text_residual_coupling_20260920'
BEST_ADAPTER = Path('/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022')
sys.path.insert(0, str(RES))
import workflow as w
from diagnostics import diagnose


def legacy():
    return w.legacy_module(HERE.parent / 'staged_coupling_20260917')


def check_predictions(rows, expected, mode, permutation):
    assert len(rows) <= len(expected)
    for r, e in zip(rows, expected):
        assert (r['id'], r['reference'], r['input'], r['condition']) == (e['id'], e['reference'], e['input'], mode)
        assert r['audio_id'] == (permutation[e['id']] if mode == 'mismatch' else e['id'])


def score(p, rows):
    m = p.metrics(rows, protected=False)
    assert (m['summary']['n'], m['summary']['chars'], m['summary']['mentions']) == (1334, 21102, 599)
    m['error_diagnostics'] = diagnose(rows, p.evaluator().norm)
    assert sum(m['error_diagnostics'][k] for k in ('S', 'D', 'I')) == m['summary']['errors']
    return m


def prepare():
    assert not (HERE / 'protocol.json').exists()
    a, p = w.configuration(OLD / 'runs/shard_mlp')
    tr, dv = p.load(a['train']), p.load(a['dev'])
    assert (len(tr), len(dv), a['steps'], a['accumulation'], a['lr'], a['seed']) == (17294, 1334, 2162, 8, 5e-5, 917)
    assert (a['warmup'], a['weight_decay'], a['max_sequence'], a['max_new_tokens']) == (32, .01, 4096, 96)
    assert not (Path(a['mlp_initialization']) / 'adapter').exists()
    p.unique(tr, 'id'); p.unique(dv, 'id')
    assert all(r['split'] == 'train' for r in tr) and all(r['split'] == 'dev' for r in dv)
    assert not ({r['id'] for r in tr} & {r['id'] for r in dv})
    norm = p.evaluator().norm
    assert not ({norm(r['reference']) for r in tr} & {norm(r['reference']) for r in dv})
    ar = OLD / 'runs/shard_mlp'
    assert w.read(ar / 'status.json')['status'] == 'COMPLETED_SHARD_DEV'
    meta = w.read(ar / 'final/metadata.json')
    assert meta['complete'] and meta['step'] == 2162 and meta['examples'] == 17294
    assert meta['protocol_sha256'] == a['protocol_sha256']
    assert meta['connector_sha256'] == w.sha(ar / 'final/connector.pt')
    am = {}
    for mode in ('text', 'true', 'zero', 'mismatch'):
        rows = p.load(ar / f'dev/{mode}.predictions.jsonl')
        check_predictions(rows, dv, mode, a['audio_permutation'])
        am[mode] = score(p, rows)
        assert am[mode] == w.read(ar / f'dev/{mode}.metrics.json')
    cache_path = ar / 'cache_manifest.json'
    cache = w.read(cache_path)
    # CPU-only sequence audit includes every row; only train rows choose smoke inputs.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p.BASE, trust_remote_code=True)
    lengths = {}
    for split, rows in (('train', tr), ('dev', dv)):
        values = []
        for r in rows:
            prompt = tok.apply_chat_template(p.messages(r, 'B', True), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            before, after = prompt.split(p.MARKER)
            n = len(tok(before, add_special_tokens=False).input_ids) + len(tok(after, add_special_tokens=False).input_ids)
            n += math.ceil(cache[r['id']]['frames'] / 4)
            target = json.dumps({'text': r['reference']}, ensure_ascii=False) + tok.eos_token
            target_n = len(tok(target, add_special_tokens=False).input_ids)
            assert n + (target_n if split == 'train' else 0) <= 4096, r['id']
            values.append((n + target_n, r['id']))
        lengths[split] = dict(max_training_style_tokens=max(values)[0], longest_id=max(values)[1])
    assert (BEST_ADAPTER / 'adapter_config.json').is_file()
    assert (BEST_ADAPTER / 'adapter_model.safetensors').is_file()
    paths = [Path(x) for x in a['sources']]
    paths += [Path(__file__), HERE / 'test_experiment.py', ar / 'protocol.json', cache_path,
              BEST_ADAPTER / 'adapter_config.json', BEST_ADAPTER / 'adapter_model.safetensors',
              ar / 'final/connector.pt', ar / 'final/metadata.json', ar / 'selected_checkpoint.json',
              *sorted((ar / 'dev').glob('*.predictions.jsonl')), *sorted((ar / 'dev').glob('*.metrics.json')),
              p.ROOT / 'cbwhisper_covo_migration_20260609_tar_extracted/covo/src/covo/text.py',
              p.ROOT / 'cbwhisper_covo_migration_20260609_tar_extracted/covo/src/covo/metrics.py']
    c = dict(a)
    c.pop('protocol_sha256', None)
    c.update(version=2, experiment='Old-best checkpoint30022 LoRA and acoustic connector co-adaptation',
             adapter=str(BEST_ADAPTER), adapter_selection='User requested historical AISHELL dev-CER-selected checkpoint30022; not claimed recall-optimal',
             sources={str(x.resolve()): w.sha(x) for x in paths}, cache_manifest=str(cache_path),
             correct_train_inputs=sum(norm(r['reference']) == norm(r['input']['asr_top1']) for r in tr),
             incorrect_train_inputs=sum(norm(r['reference']) != norm(r['input']['asr_top1']) for r in tr),
             groups={'A': 'Afull MLP + frozen checkpoint30022, newly train connector only',
                     'B': 'Afull MLP + checkpoint30022 LoRA, jointly train both',
                     'C': 'checkpoint30022 LoRA only, no acoustic input',
                     'D': 'Unchanged checkpoint30022 matched-prompt text baseline, no training'},
             group_A=str(HERE / 'A'), lora_lr=5e-6, lora_config=w.read(BEST_ADAPTER / 'adapter_config.json'),
             trainable='A connector only; B connector+existing LoRA; C existing LoRA; D none; all base parameters frozen',
             selection='Fixed final 2162 updates, all 17294 rows once; no dev selection',
             checkpoint_interval=256, sequence_audit=lengths,
             smoke_ids=[tr[0]['id'], lengths['train']['longest_id']],
             comparison='A/B/C same examples/order/update count, not equal FLOPs. All start checkpoint30022. A/B share inherited Afull trained with625. B/C retain existing LoRA dropout.',
             objective='Reference JSON CE only; no extra alignment or preservation losses; no output protection',
             evaluation={'A': ['text', 'true', 'zero', 'mismatch'], 'B': ['text', 'true', 'zero', 'mismatch'], 'C': ['text'], 'D': ['text']},
             primary='CER and designated hotword recall; report all conditions and paired 95% bootstrap CIs',
             success='B true no higher CER and higher recall than D text; superiority vs C and correct-audio controls examined separately',
             limitations=['Previously examined 1334 contextual dev, not official full dev or held-out test',
                          'Historical frontend/context inherited; Afull was pretrained with checkpoint625, now transferred to30022 equally in A/B',
                          'Historical dev-CER selection of30022 is not evidence of optimal contextual recall',
                          'Error-enriched training retained to isolate adaptation; no causal architecture claim',
                          'Single seed; mismatch has imperfect frame-length matching',
                          'Point non-degradation is not a statistical noninferiority guarantee'])
    w.save(HERE / 'protocol.json', c)
    w.save(HERE / 'historical_625_audit_NOT_CURRENT_CONTROL.json', dict(protocol_sha256=a['protocol_sha256'], summaries={k:v['summary'] for k,v in am.items()},
                                           verified='Historical data/cache/schedule provenance only. Predictions are NOT reused as30022 controls.'))
    for g in ('A', 'B', 'C', 'D'):
        (HERE / g).mkdir(exist_ok=False)
    w.save(HERE / 'status.json', dict(status='PREPARED', train_n=len(tr), dev_n=len(dv)))
    print(json.dumps(dict(status='PREPARED', sequence_audit=lengths, correct_train_inputs=c['correct_train_inputs'])), flush=True)


def configuration():
    c = w.read(HERE / 'protocol.json')
    for name, expected in c['sources'].items():
        assert w.sha(name) == expected, f'Locked source changed: {name}'
    assert w.sha(c['cache_manifest']) == c['cache_manifest_sha256']
    for info in w.read(c['cache_manifest']).values():
        assert w.sha(info['path']) == info['sha256'], info['path']
    c['protocol_sha256'] = w.sha(HERE / 'protocol.json')
    p = legacy()
    assert c['adapter'] == str(BEST_ADAPTER), 'Do not silently fall back to checkpoint625'
    p.ADAPTER = Path(c['adapter'])
    return c, p


class Runtime:
    def __init__(self, c, p, group, training, checkpoint=None):
        self.c, self.p, self.group = c, p, group
        # Stage A freezes the model/LoRA; B enables only the existing LoRA.
        self.model, self.tokenizer, self.connector = p.load_model('A' if group == 'A' else 'B', checkpoint or c['mlp_initialization'], training)
        if group in ('C', 'D'):
            self.connector.requires_grad_(False)
        self.lora = [(n, x) for n, x in self.model.named_parameters() if x.requires_grad]
        if training:
            assert bool(self.lora) == (group in ('B', 'C'))
            assert all('lora_' in n for n, _ in self.lora)
            self.model.train()
            self.connector.train(group in ('A', 'B'))
        else:
            assert not self.lora
        self.parameters = ([x for x in self.connector.parameters() if x.requires_grad] if group in ('A', 'B') else []) + [x for _, x in self.lora]

    def batch(self, row, training=False, mode=None, audio_id=None):
        return self.p.batch(self.model, self.tokenizer, self.connector, row, 'B',
                            mode or ('true' if self.group in ('A', 'B') else 'text'), audio_id, training)

    def optimizer(self):
        import torch
        groups = []
        if self.group in ('A', 'B'):
            groups.append(dict(params=list(self.connector.parameters()), lr=self.c['lr']))
        if self.lora:
            groups.append(dict(params=[x for _, x in self.lora], lr=self.c['lora_lr']))
        opt = torch.optim.AdamW(groups, weight_decay=self.c['weight_decay'])
        c = self.c
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/c['warmup'] if s<c['warmup'] else max(0., (c['steps']-s)/(c['steps']-c['warmup'])))
        return opt, sch


def update(rt, chunk, opt, sch):
    import torch
    opt.zero_grad(set_to_none=True)
    losses = []
    for row in chunk:
        b = rt.batch(row, training=True)
        labels = b['labels'][0]
        mask = labels.ne(-100)
        assert mask.any() and not mask[0] and mask[-1]
        assert torch.equal(labels[mask], rt.tokenizer(json.dumps({'text':row['reference']}, ensure_ascii=False)+rt.tokenizer.eos_token,
                                                     add_special_tokens=False, return_tensors='pt').input_ids[0].to(labels.device))
        loss = rt.model(**b, use_cache=False).loss
        assert torch.isfinite(loss)
        (loss / len(chunk)).backward()
        losses.append(float(loss.detach()))
    assert all(x.grad is None for x in rt.model.parameters() if not x.requires_grad)
    lora_grad = sum(float(x.grad.detach().float().norm()) for _, x in rt.lora if x.grad is not None)
    conn_grad = sum(float(x.grad.detach().float().norm()) for x in rt.connector.parameters() if x.grad is not None)
    assert (math.isfinite(lora_grad) and lora_grad > 0) if rt.group in ('B', 'C') else lora_grad == 0
    assert (math.isfinite(conn_grad) and conn_grad > 0) if rt.group in ('A', 'B') else conn_grad == 0
    grad = torch.nn.utils.clip_grad_norm_(rt.parameters, 1.)
    assert torch.isfinite(grad) and grad > 0
    opt.step(); sch.step()
    return dict(loss=sum(losses)/len(losses), grad_norm=float(grad), lora_grad=lora_grad, connector_grad=conn_grad)


def checkpoint(rt, opt, sch, path, step, examples):
    import numpy as np
    import torch
    path.mkdir(exist_ok=False)
    torch.save(rt.connector.state_dict(), path / 'connector.pt')
    rt.model.save_pretrained(path / 'adapter')
    torch.save(dict(optimizer=opt.state_dict(), scheduler=sch.state_dict(), step=step,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), path / 'resume.pt')
    files = {str(q.relative_to(path)):w.sha(q) for q in path.rglob('*') if q.is_file()}
    w.save(path / 'metadata.json', dict(complete=True, group=rt.group, step=step, examples=examples,
                                       protocol_sha256=rt.c['protocol_sha256'], hashes=files))
    return path


def check_checkpoint(path, c, group):
    meta = w.read(path / 'metadata.json')
    assert meta['complete'] and meta['protocol_sha256'] == c['protocol_sha256'] and meta['group'] == group
    for name, expected in meta['hashes'].items():
        assert w.sha(path / name) == expected, name
    return meta


def restore(rt, opt, sch, path):
    import numpy as np
    import torch
    meta = check_checkpoint(path, rt.c, rt.group)
    # This is our own hash-verified optimizer/RNG checkpoint, not third-party pickle.
    state = torch.load(path / 'resume.pt', map_location='cpu', weights_only=False)
    assert state['step'] == meta['step']
    opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    return meta


def smoke(group, reload_only=False):
    import torch
    from peft import get_peft_model_state_dict
    c, p = configuration(); d = HERE / group
    dest = d / 'smoke_checkpoint'
    with w.gpu_guard(p):
        rt = Runtime(c, p, group, True, dest if reload_only else None)
        opt, sch = rt.optimizer()
        rows = {r['id']:r for r in p.load(c['train'])}
        if reload_only:
            meta = restore(rt, opt, sch, dest)
            assert meta['step'] == 2 and sch.last_epoch == 2
            from safetensors.torch import load_file
            saved = load_file(str(dest / 'adapter/adapter_model.safetensors'))
            loaded = get_peft_model_state_dict(rt.model)
            assert saved.keys() == loaded.keys()
            assert all(torch.equal(v, loaded[k].cpu()) for k,v in saved.items())
            cs = torch.load(dest / 'connector.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(v, rt.connector.state_dict()[k].cpu()) for k,v in cs.items())
            assert all(int(v['step']) == 2 for v in opt.state.values())
            rt.model.eval(); rt.connector.eval()
            pr = p.generate(rt.model, rt.tokenizer, rt.connector, rows[c['smoke_ids'][0]], 'B', 'true' if group in ('A','B') else 'text')
            assert pr['model_output'] == w.read(d / 'smoke_before_reload.json')['model_output']
            rt.model.train(); rt.connector.train(group in ('A','B'))
            rec = update(rt, [rows[c['smoke_ids'][0]]], opt, sch)
            assert sch.last_epoch == 3
            w.save(d / 'smoke.json', dict(status='PASS', group=group, reload_exact=True, resumed_update=rec,
                                         protocol_sha256=c['protocol_sha256'], max_gpu_gb=torch.cuda.max_memory_allocated()/1e9))
        else:
            assert not dest.exists()
            before = {n:x.detach().cpu().clone() for n,x in rt.lora}
            cbefore = {n:x.detach().cpu().clone() for n,x in rt.connector.named_parameters()}
            frozen_versions = {n:x._version for n,x in rt.model.named_parameters() if not x.requires_grad}
            recs = []
            for uid in c['smoke_ids']:
                began = time.time(); rec = update(rt, [rows[uid]], opt, sch)
                rec.update(id=uid, seconds=time.time()-began); recs.append(rec)
                print(json.dumps(dict(smoke=group, **rec)), flush=True)
            assert any(not torch.equal(before[n], x.cpu()) for n,x in rt.lora) == (group in ('B','C'))
            changed = any(not torch.equal(cbefore[n], x.cpu()) for n,x in rt.connector.named_parameters())
            assert changed == (group in ('A','B'))
            assert all(x._version == frozen_versions[n] for n,x in rt.model.named_parameters() if not x.requires_grad)
            checkpoint(rt, opt, sch, dest, 2, 2)
            rt.model.eval(); rt.connector.eval()
            pr = p.generate(rt.model, rt.tokenizer, rt.connector, rows[c['smoke_ids'][0]], 'B', 'true' if group in ('A','B') else 'text')
            w.save(d / 'smoke_before_reload.json', pr)
            w.save(d / 'smoke_training.json', dict(updates=recs, llm_trainable=sum(x.numel() for _,x in rt.lora),
                                                   max_gpu_gb=torch.cuda.max_memory_allocated()/1e9))
        print(json.dumps(dict(smoke=group, reload=reload_only, status='PASS')), flush=True)


def train(group):
    import torch
    c, p = configuration(); d=HERE/group
    assert w.read(d/'smoke.json')['status'] == 'PASS'
    assert w.read(d/'smoke.json')['protocol_sha256'] == c['protocol_sha256']
    if (d/'selected_checkpoint.json').exists():
        sel=w.read(d/'selected_checkpoint.json'); meta=check_checkpoint(Path(sel['checkpoint']),c,group)
        assert meta['step']==c['steps'] and meta['examples']==c['train_n']; return
    previous=[]
    for q in (d/'checkpoints').glob('step_*'):
        if (q/'metadata.json').exists():
            meta=check_checkpoint(q,c,group); previous.append((meta['step'],q))
    start, cp=max(previous,default=(0,None))
    with w.gpu_guard(p):
        rt=Runtime(c,p,group,True,cp); opt,sch=rt.optimizer()
        rows=p.load(c['train']); random.Random(c['seed']).shuffle(rows)
        if cp: restore(rt,opt,sch,cp)
        run_id=time.time_ns(); began=time.time()
        w.save(d/f'training_started_{run_id}.json',dict(pid=os.getpid(),started=began,resume_step=start,protocol_sha256=c['protocol_sha256']))
        (d/'checkpoints').mkdir(exist_ok=True)
        with (d/f'training_{run_id}.jsonl').open('x',encoding='utf-8') as stream:
            for step in range(start,c['steps']):
                chunk=rows[step*c['accumulation']:(step+1)*c['accumulation']]
                rec=update(rt,chunk,opt,sch)
                rec.update(status='TRAINING',group=group,step=step+1,steps=c['steps'],examples=min((step+1)*c['accumulation'],len(rows)),
                           seconds=time.time()-began,resume_step=start,max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
                stream.write(json.dumps(rec)+'\n'); stream.flush(); w.save(d/'status.json',rec)
                if step==start or (step+1)%10==0: print(json.dumps(rec),flush=True)
                if (step+1)%c['checkpoint_interval']==0 or step+1==c['steps']:
                    q=d/'checkpoints'/f'step_{step+1}_{time.time_ns()}'
                    checkpoint(rt,opt,sch,q,step+1,rec['examples'])
                    w.save(d/'latest_checkpoint.json',dict(checkpoint=str(q),step=step+1))
        w.save(d/'selected_checkpoint.json',dict(checkpoint=str(q),step=c['steps'],reason=c['selection']))
        w.save(d/'status.json',dict(status='TRAINED_NOT_EVALUATED',checkpoint=str(q)))


def evaluate(group):
    c,p=configuration(); d=HERE/group
    if group=='D':
        cp=None
    else:
        cp=Path(w.read(d/'selected_checkpoint.json')['checkpoint']); meta=check_checkpoint(cp,c,group)
        assert meta['step']==2162 and meta['examples']==17294
    rows=p.load(c['dev']); dest=d/'dev'; dest.mkdir(exist_ok=True)
    modes=c['evaluation'][group]
    missing=[m for m in modes if not (dest/f'{m}.predictions.jsonl').exists()]
    if missing:
        with w.gpu_guard(p):
            rt=Runtime(c,p,group,False,cp)
            for mode in missing:
                target=dest/f'{mode}.predictions.jsonl'; partial=Path(str(target)+'.partial')
                prior=p.load(partial) if partial.exists() else []
                check_predictions(prior,rows,mode,c['audio_permutation'])
                with partial.open('a',encoding='utf-8') as stream:
                    for i in range(len(prior),len(rows)):
                        r=rows[i]
                        pr=p.generate(rt.model,rt.tokenizer,rt.connector,r,'B',mode,c['audio_permutation'][r['id']] if mode=='mismatch' else None)
                        stream.write(json.dumps(pr,ensure_ascii=False)+'\n'); stream.flush()
                        if (i+1)%25==0 or i+1==len(rows):
                            rec=dict(status='EVALUATING',group=group,mode=mode,done=i+1,total=len(rows))
                            w.save(d/'status.json',rec); print(json.dumps(rec),flush=True)
                rr=p.load(partial); check_predictions(rr,rows,mode,c['audio_permutation']); assert len(rr)==len(rows)
                partial.rename(target)
    summaries={}
    for mode in modes:
        rr=p.load(dest/f'{mode}.predictions.jsonl'); check_predictions(rr,rows,mode,c['audio_permutation'])
        m=score(p,rr); w.save(dest/f'{mode}.metrics.json',m); summaries[mode]=m['summary']
    w.save(dest/'summary.json',dict(group=group,summaries=summaries,protocol_sha256=c['protocol_sha256'],checkpoint=str(cp) if cp else c['adapter'],no_output_repairs=True))
    w.save(d/'status.json',dict(status='COMPLETE',summary=str(dest/'summary.json')))


def report():
    c,p=configuration(); rows=p.load(c['dev']); metrics={}; hashes={}
    for group,folder,modes in [(g,HERE/g,c['evaluation'][g]) for g in ('A','B','C','D')]:
        for mode in modes:
            path=folder/'dev'/f'{mode}.predictions.jsonl'; rr=p.load(path)
            check_predictions(rr,rows,mode,c['audio_permutation']); metrics[f'{group}_{mode}']=score(p,rr); hashes[f'{group}_{mode}']=w.sha(path)
    comparisons={name:p.bootstrap(metrics['B_true']['per_utterance'],metrics[name]['per_utterance'])
                 for name in ('D_text','A_text','A_true','B_text','B_zero','B_mismatch','C_text')}
    original=score(p,[dict(r,prediction=r['input']['asr_top1'],parsed=True) for r in rows])
    result=dict(summaries={k:v['summary'] for k,v in metrics.items()},input_summary=original['summary'],
                diagnostics={k:v['error_diagnostics'] for k,v in metrics.items()},B_true_minus_control_95ci=comparisons,
                protocol_sha256=c['protocol_sha256'],prediction_hashes=hashes,no_output_repairs=True,limitations=c['limitations'])
    w.save(HERE/'summary.json',result)
    w.save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    import zipfile
    with zipfile.ZipFile(HERE/f'audited_results_{time.time_ns()}.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for q in HERE.rglob('*'):
            if q.is_file() and q.suffix in ('.json','.jsonl','.py','.md') and 'adapter' not in q.parts:
                z.write(q,q.relative_to(HERE))
    print(json.dumps(result),flush=True)


def pipeline():
    import fcntl
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configuration()
        tasks=[('smoke','A'),('reload-smoke','A'),('smoke','B'),('reload-smoke','B'),('smoke','C'),('reload-smoke','C'),
               ('eval','D'),('train','B'),('eval','B'),('train','C'),('eval','C'),('train','A'),('eval','A'),('report',)]
        for task in tasks:
            if task[0] in ('smoke','reload-smoke') and (HERE/task[1]/'smoke.json').exists():
                assert w.read(HERE/task[1]/'smoke.json')['status']=='PASS'; continue
            if task[0]=='smoke' and (HERE/task[1]/'smoke_checkpoint/metadata.json').exists(): continue
            w.save(HERE/'status.json',dict(status='RUNNING',phase=list(task),pid=os.getpid(),updated=time.time()))
            log=HERE/('_'.join(task)+f'_{time.time_ns()}.log')
            with log.open('x') as stream:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),*task],stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert w.read(HERE/'status.json')['status']=='ALL_DONE'


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','smoke','reload-smoke','train','eval','report','pipeline']); ap.add_argument('group',nargs='?',choices=['A','B','C','D']); args=ap.parse_args()
    try:
        if args.command=='prepare': prepare()
        elif args.command=='smoke': smoke(args.group)
        elif args.command=='reload-smoke': smoke(args.group,True)
        elif args.command=='train': train(args.group)
        elif args.command=='eval': evaluate(args.group)
        elif args.command=='report': report()
        else: pipeline()
    except BaseException:
        err=dict(status='FAILED',command=vars(args),error=traceback.format_exc(),time=time.time())
        w.save(HERE/f'failure_{time.time_ns()}.json',err); w.save(HERE/'status.json',err); raise
