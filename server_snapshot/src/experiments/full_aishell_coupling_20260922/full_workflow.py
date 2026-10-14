"""One complete epoch and complete official-dev validation. No subset fallback.

Requires frozen, complete CBSense evidence before prepare can succeed.
Reuses audited connector implementation without modifying the pilot protocol.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RESIDUAL = HERE.parent / 'text_residual_coupling_20260920'
sys.path.insert(0, str(RESIDUAL))
import workflow as w
from prepare_full import legacy, DATA


def prepare(architecture):
    p = legacy()
    audit = w.read(DATA / 'audit.json')
    ready = w.read(DATA / 'evidence_complete.json')
    rows = {s: p.load(DATA / f'{s}.jsonl') for s in ('train','dev')}
    ev = p.evaluator()
    for s, rr in rows.items():
        expected = p.load(DATA / f'{s}.manifest.jsonl')
        assert len(rr) == audit[f'{s}_n']
        assert [r['id'] for r in rr] == [r['id'] for r in expected]
        assert all(r['split'] == s and r['reference'] == m['reference'] and r['wav'] == m['wav']
                   for r,m in zip(rr,expected))
        assert w.sha(DATA / f'{s}.jsonl') == ready['sha256'][s]
        assert all(ev.norm(r['input']['asr_top1']) for r in rr), 'Empty ASR inputs require diagnosis, not dropping rows'
    assert not {ev.norm(r['reference']) for r in rows['train']} & {ev.norm(r['reference']) for r in rows['dev']}
    config = w.read(RESIDUAL / 'runs' / ('residual_v1' if architecture == 'residual' else 'mlp_B_control') / 'protocol.json')
    dest = HERE / 'runs' / architecture
    dest.mkdir(parents=True, exist_ok=False)
    # The source implementation remains unchanged; all dataset/budget fields are replaced.
    config.update(version=2, train=str(DATA / 'train.jsonl'), dev=str(DATA / 'dev.jsonl'),
                  train_n=len(rows['train']), dev_n=len(rows['dev']),
                  steps=math.ceil(len(rows['train']) / 8), accumulation=8,
                  selection='Fixed final checkpoint after exactly one complete epoch; no dev-based tuning',
                  label=audit['label'], data_caveat=audit['train_policy'],
                  external_context_caveat=ready['context_policy'],
                  historical_adapter_caveat=audit['historical_adapter_caveat'],
                  historical_Afull_manifest_dev_reference_overlap=audit['historical_Afull_manifest_dev_reference_overlap'],
                  correct_train_inputs=sum(ev.norm(r['reference']) == ev.norm(r['input']['asr_top1']) for r in rows['train']),
                  audio_permutation=p.permutation(rows['dev']))
    config['incorrect_train_inputs'] = config['train_n'] - config['correct_train_inputs']
    sources = [Path(q) for q in config['sources'] if q not in {config.get('train'), config.get('dev')}]
    sources += [Path(__file__), HERE / 'prepare_full.py', DATA / 'audit.json', DATA / 'evidence_complete.json',
                DATA / 'train.jsonl', DATA / 'dev.jsonl', p.ALIGNED.parent / 'hotword.txt']
    config['sources'] = {str(q):w.sha(q) for q in sources}
    cache = {}
    import numpy as np
    for i,r in enumerate(rows['train'] + rows['dev']):
        q = w.audio_path(p,r['id'])
        x = np.load(q,mmap_mode='r',allow_pickle=False)
        assert x.ndim == 2 and x.shape[1] == 512 and len(x) and np.isfinite(x).all()
        cache[r['id']] = dict(path=str(q),sha256=w.sha(q),frames=len(x))
        if (i+1) % 5000 == 0:
            print(json.dumps(dict(cache_audited=i+1)),flush=True)
    w.save(dest / 'cache_manifest.json',cache)
    config['cache_manifest_sha256'] = w.sha(dest / 'cache_manifest.json')
    w.save(dest / 'protocol.json',config)
    w.save(dest / 'status.json',dict(status='PREPARED_FULL_DATA',training='NOT_STARTED'))


def train(directory):
    import torch
    directory = Path(directory)
    c,p = w.configuration(directory)
    smoke = w.read(directory / 'gpu_smoke.json')
    assert smoke['status'] == 'PASS' and smoke['protocol_sha256'] == c['protocol_sha256']
    assert not (directory / 'final').exists(), 'Already trained; never overwrite'
    with w.gpu_guard(p):
        runtime = w.Runtime(c,p,training=True)
        rows = p.load(c['train'])
        assert len(rows) == c['train_n'] == 119030 and c['steps'] == math.ceil(len(rows)/c['accumulation'])
        random.Random(c['seed']).shuffle(rows)
        optimizer = torch.optim.AdamW(runtime.connector.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s: (s+1)/c['warmup'] if s < c['warmup']
                     else max(0.,(c['steps']-s)/(c['steps']-c['warmup'])))
        step0 = 0
        resume = directory / 'resume.json'
        if resume.exists():
            saved = w.read(resume)
            assert saved['protocol_sha256'] == c['protocol_sha256']
            assert w.sha(saved['path']) == saved['sha256']
            state = torch.load(saved['path'],map_location='cpu',weights_only=False)
            runtime.connector.load_state_dict(state['connector'])
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
            step0 = state['step']
        started = time.time()
        attempt = directory / f'training_attempt_{time.time_ns()}.jsonl'
        w.save(directory / 'training_started.json',dict(pid=os.getpid(),start_step=step0,started=started,log=str(attempt)))
        with attempt.open('x') as log:
            for step in range(step0,c['steps']):
                chunk = rows[step*c['accumulation']:(step+1)*c['accumulation']]
                assert chunk
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for row in chunk:
                    loss = runtime.model(**runtime.batch(row,training=True),use_cache=False).loss
                    assert torch.isfinite(loss), row['id']
                    (loss/len(chunk)).backward()
                    loss_sum += float(loss.detach())
                grad = torch.nn.utils.clip_grad_norm_(runtime.connector.parameters(),1.)
                assert torch.isfinite(grad) and grad > 0
                assert all(q.grad is None for q in runtime.model.parameters())
                optimizer.step(); scheduler.step()
                record = dict(status='TRAINING_FULL',step=step+1,steps=c['steps'],examples=min((step+1)*c['accumulation'],len(rows)),
                              loss=loss_sum/len(chunk),grad_norm=float(grad),seconds=time.time()-started)
                log.write(json.dumps(record)+'\n'); log.flush()
                w.save(directory / 'status.json',record)
                if (step+1) % 10 == 0 or step == step0:
                    print(json.dumps(record),flush=True)
                if (step+1) % 512 == 0 or step+1 == c['steps']:
                    path = directory / f'resume_step_{step+1:06d}.pt'
                    assert not path.exists()
                    temp = Path(str(path)+'.partial')
                    torch.save(dict(step=step+1,connector=runtime.connector.state_dict(),optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
                    temp.replace(path)
                    w.save(resume,dict(path=str(path),sha256=w.sha(path),protocol_sha256=c['protocol_sha256']))
        final = directory / 'final'
        final.mkdir(exist_ok=False)
        torch.save(runtime.connector.state_dict(),final / 'connector.pt')
        w.save(final / 'metadata.json',dict(architecture=c['architecture'],step=c['steps'],examples=len(rows),
               protocol_sha256=c['protocol_sha256'],connector_sha256=w.sha(final / 'connector.pt'),complete=True))
        w.save(directory / 'selected_checkpoint.json',dict(checkpoint=str(final),connector_sha256=w.sha(final / 'connector.pt'),reason=c['selection']))
        w.save(directory / 'status.json',dict(status='FULL_EPOCH_TRAINED_NOT_EVALUATED'))


def evaluate(directory):
    from diagnostics import diagnose
    directory = Path(directory)
    c,p = w.configuration(directory)
    selected = w.read(directory / 'selected_checkpoint.json')
    assert selected['connector_sha256'] == w.sha(directory / 'final/connector.pt')
    rows = p.load(c['dev'])
    assert len(rows) == c['dev_n'] == 14326
    norm = p.evaluator().norm
    vocab = sorted({norm(x) for x in (p.ALIGNED.parent / 'hotword.txt').read_text().splitlines() if norm(x)})
    # Labels used for scoring only; never attached to inference evidence or routing.
    dest = directory / 'official_full_dev'
    dest.mkdir(exist_ok=True)
    aligned = dest / 'fixed600_term_presence_pairs.tsv'
    annotation = ''.join(f'{t}\t{r["id"]}\n' for r in rows for t in vocab if t in norm(r['reference']))
    if aligned.exists():
        assert aligned.read_text() == annotation
    else:
        aligned.write_text(annotation)
    p.ALIGNED = aligned
    with w.gpu_guard(p):
        runtime = w.Runtime(c,p,checkpoint=directory / 'final')
        all_metrics = {}
        for mode in c['modes']:
            path = dest / f'{mode}.predictions.jsonl'
            partial = Path(str(path)+'.partial')
            existing = p.load(path if path.exists() else partial) if path.exists() or partial.exists() else []
            assert len(existing) <= len(rows)
            assert all(a['id'] == b['id'] and a['reference'] == b['reference'] and a['input'] == b['input']
                       and a['condition'] == mode for a,b in zip(existing,rows))
            if path.exists():
                assert len(existing) == len(rows)
            else:
                with partial.open('a') as out:
                    for i in range(len(existing),len(rows)):
                        row = rows[i]
                        pred = runtime.predict(row,mode,c['audio_permutation'][row['id']] if mode == 'mismatch' else None)
                        out.write(json.dumps(pred,ensure_ascii=False)+'\n'); out.flush()
                        if (i+1) % 50 == 0:
                            progress = dict(status='EVALUATING_OFFICIAL_FULL_DEV',mode=mode,done=i+1,total=len(rows))
                            w.save(directory / 'status.json',progress)
                            print(json.dumps(progress),flush=True)
                partial.rename(path)
            predictions = p.load(path)
            assert len(predictions) == len(rows)
            metric = p.metrics(predictions,protected=False)
            assert metric['summary']['chars'] == 205341
            metric['error_diagnostics'] = diagnose(predictions,norm)
            assert sum(metric['error_diagnostics'].get(k,0) for k in ('S','D','I')) == metric['summary']['errors']
            metric['recall_definition'] = 'Unique (utterance, fixed-600-vocabulary term) presence pairs; NOT historical 599 annotated mentions'
            w.save(dest / f'{mode}.metrics.json',metric)
            all_metrics[mode] = metric
        assert [r['model_output'] for r in p.load(dest / 'off.predictions.jsonl')] == [r['model_output'] for r in p.load(dest / 'text.predictions.jsonl')]
        w.save(dest / 'summary.json',dict(label=c['label'],summaries={m:q['summary'] for m,q in all_metrics.items()},
               paired_bootstrap={m:p.bootstrap(all_metrics['true']['per_utterance'],all_metrics[m]['per_utterance']) for m in ('text','zero','mismatch')},
               no_output_repairs=True,normalization=c['normalization'],historical_adapter_caveat=c['historical_adapter_caveat'],
               objective='Lower CER without lower contextual recall; point estimates and paired confidence intervals, not a guarantee',
               predictions_sha256={m:w.sha(dest/f'{m}.predictions.jsonl') for m in c['modes']}))
        w.save(directory / 'status.json',dict(status='COMPLETED_OFFICIAL_FULL_DEV',n=len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','smoke','train','eval'])
    parser.add_argument('architecture',choices=['residual','mlp'])
    args = parser.parse_args()
    directory = HERE / 'runs' / args.architecture
    if args.command == 'prepare':
        prepare(args.architecture)
    else:
        try:
            {'smoke':w.smoke,'train':train,'eval':evaluate}[args.command](directory)
        except BaseException:
            w.save(directory / f'failure_{time.time_ns()}.json',dict(command=args.command,error=traceback.format_exc()))
            raise
