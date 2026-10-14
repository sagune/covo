"""Diagnostic CTC syllable probes on frozen SenseVoice caches; no 9B loading."""
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

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'joint_lora_20260923'
sys.path.insert(0, str(OLD))
import experiment as prior
w = prior.w


def syllables(text, norm):
    """Strict spelling-derived, toneless syllables; never guess numeral readings."""
    text = norm(text)
    if not text or not all('\u4e00' <= x <= '\u9fff' for x in text):
        return None
    from pypinyin import lazy_pinyin, Style
    result = lazy_pinyin(text, style=Style.NORMAL)
    if any(not x.isascii() or not x.isalpha() for x in result):
        return None
    assert len(result) == len(text)
    return result


def align(ref, hyp):
    """Minimum edit alignment, diagonal/deletion/insertion tie breaking."""
    n, m = len(ref), len(hyp)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (ref[i-1] != hyp[j-1]), dp[i-1][j] + 1, dp[i][j-1] + 1)
    positions = {}; edits = dict(S=0, D=0, I=0); kinds = {}; i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            positions[i-1] = hyp[j-1]
            kinds[i-1] = 'M' if ref[i-1] == hyp[j-1] else 'S'
            edits['S'] += ref[i-1] != hyp[j-1]; i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            positions[i-1] = None; kinds[i-1] = 'D'; edits['D'] += 1; i -= 1
        else:
            edits['I'] += 1; j -= 1
    assert sum(edits.values()) == dp[n][m]
    return dict(errors=dp[n][m], edits=edits, positions=positions, kinds=kinds)


def greedy(indices, vocabulary):
    result = []; last = None
    for index in indices:
        if index != last and index != 0:
            result.append(vocabulary[index])
        last = index
    return result


def min_ctc_frames(target):
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def prepare():
    import numpy as np
    import pypinyin
    assert not (HERE / 'protocol.json').exists(), 'Immutable experiment already prepared'
    old, p = prior.configuration(); norm = p.evaluator().norm
    train, dev = p.load(old['train']), p.load(old['dev'])
    assert (len(train), len(dev)) == (17294, 1334)
    p.unique(train, 'id'); p.unique(dev, 'id')
    assert all(r['split'] == 'train' for r in train) and all(r['split'] == 'dev' for r in dev)
    assert not ({r['id'] for r in train} & {r['id'] for r in dev})
    assert not ({norm(r['reference']) for r in train} & {norm(r['reference']) for r in dev})
    cache = w.read(old['cache_manifest'])
    targets = {}; audit = {}
    for split, rows in [('train', train), ('dev', dev)]:
        targets[split] = []
        for row in rows:
            phones = syllables(row['reference'], norm)
            # No silent removal if the preliminary audit differs.
            assert phones is not None, ('Unsupported reference', row['id'], row['reference'])
            assert cache[row['id']]['frames'] >= min_ctc_frames(phones), row['id']
            targets[split].append(dict(id=row['id'], split=split, reference=row['reference'],
                                       syllables=phones, input=row['input']))
        audit[split] = dict(n=len(rows), syllables=sum(len(x['syllables']) for x in targets[split]),
                            unsupported_reference=0)
        p.save_rows(HERE / (split + '_targets.jsonl'), targets[split])
    vocab = ['<blank>'] + sorted({x for r in targets['train'] for x in r['syllables']})
    w.save(HERE / 'vocabulary.json', vocab)
    unseen = sorted({x for r in targets['dev'] for x in r['syllables']} - set(vocab))
    audit['dev_unseen_syllables'] = unseen
    # Historical 9B outputs are diagnostic comparisons ONLY, never training inputs.
    old_text = OLD / 'B/dev/text.predictions.jsonl'
    baseline = p.load(old_text); prior.check_predictions(baseline, dev, 'text', old['audio_permutation'])
    assert len(baseline) == len(dev)
    comparisons = []
    for row, text in zip(targets['dev'], baseline):
        item = dict(id=row['id'], frontend=syllables(row['input']['asr_top1'], norm),
                    old_B_text=syllables(text['prediction'], norm),
                    frontend_text=row['input']['asr_top1'], old_B_text_output=text['prediction'])
        comparisons.append(item)
    p.save_rows(HERE / 'dev_comparisons.jsonl', comparisons)
    for key in ('frontend', 'old_B_text'):
        audit[key + '_Han_only_eligible'] = sum(r[key] is not None for r in comparisons)
    paths = [Path(__file__), HERE/'test_probe.py', HERE/'PLAN.md', HERE/'vocabulary.json',
             HERE/'train_targets.jsonl', HERE/'dev_targets.jsonl', HERE/'dev_comparisons.jsonl',
             OLD/'protocol.json', old_text]
    sources = {str(x.resolve()): w.sha(x) for x in paths}
    c = dict(version=1, experiment='Frozen SenseVoice toneless syllable identifiability probes',
             sources=sources, parent_protocol_sha256=old['protocol_sha256'],
             train_targets=str(HERE/'train_targets.jsonl'), dev_targets=str(HERE/'dev_targets.jsonl'),
             comparisons=str(HERE/'dev_comparisons.jsonl'), cache_manifest=old['cache_manifest'],
             cache_manifest_sha256=w.sha(old['cache_manifest']), vocabulary=str(HERE/'vocabulary.json'),
             seed=917, epochs=10, batch_size=32, lr=.001, weight_decay=.01, warmup=50,
             gradient_clip=1., heads=['linear','mlp'], modes=['true','zero','mismatch'],
             selection='Fixed epoch10 for both probes; no dev-based checkpoint choice or tuning',
             trainable='Probe LayerNorm and linear/MLP parameters only; 9B/LoRA/SenseVoice not loaded or updated',
             architecture='linear: LN512+Linear400; mlp: LN512+Linear256+GELU+Linear400 (vocabulary size is train derived)',
             objective='CTC mean per-target-syllable negative log likelihood, no zero_infinity masking',
             decode='Greedy CTC collapse, no beam, lexicon, hotwords or language model',
             target='pypinyin Style.NORMAL sentence-context default readings; tone-free whole syllables, not individual phonemes',
             pypinyin_version=pypinyin.__version__, audit=audit, audio_permutation=old['audio_permutation'],
             smoke_ids=[targets['train'][0]['id'], max(targets['train'],key=lambda x:cache[x['id']]['frames'])['id']],
             metrics='Corpus syllable error rate (SyER), S/D/I, raw decoding, correctly/missing/substituted baseline position diagnostics, '
                     'reference-vs-baseline CTC preference as reference-aware diagnostic, paired95% bootstrap',
             baseline_policy='Non-Han baseline strings are flagged and retained; matched phonetic comparisons use each baseline Han-only subset. '
                             'No guessed digit/English readings. Probe full-dev SyER always1334 rows/21102 syllables.',
             success='Evidence useful if true audio offers reference-pronunciation discrimination at actual errors beyond erasure/mismatch '
                     'without excessive false changes; greedy SyER need not outperform a full contextual ASR to show local information',
             interpretation='Diagnostic feature probe, not a deployed correction model or independent encoder; no word CER/recall claim',
             limitations=['Repeated historical contextualdev, not official full dev or new heldout test',
                          'Dictionary pronunciations are not waveform-verified; polyphones and numeral readings need separate checks',
                          'Same-phoneme spellings cannot be identified by this target; tone-only errors deliberately outside first probe',
                          'Simple probes can underfit; failure does not prove features contain no information',
                          'Full greedy transcription and reference-aware CTC pair preference are different diagnostic tasks',
                          'Zero is erased cached encoder frames, not re-encoded silence; mismatch has imperfect duration matching',
                          'One seed, both heads use all17294 train rows repeatedly; no train-size model search'])
    w.save(HERE/'protocol.json',c)
    for head in c['heads']:
        (HERE/head).mkdir(exist_ok=False)
    w.save(HERE/'status.json',dict(status='PREPARED',audit=audit,vocabulary_n=len(vocab)))
    print(json.dumps(dict(audit=audit,vocabulary_n=len(vocab))),flush=True)


def configuration():
    old, p = prior.configuration()  # Read-only cache and source identities, no 9B load.
    c = w.read(HERE/'protocol.json')
    assert old['protocol_sha256'] == c['parent_protocol_sha256']
    for path, expected in c['sources'].items():
        assert w.sha(path) == expected, 'Locked source changed: '+path
    assert w.sha(c['cache_manifest']) == c['cache_manifest_sha256']
    c['protocol_sha256'] = w.sha(HERE/'protocol.json')
    return c, p


def make_model(head, vocab_n):
    import torch
    layers = [torch.nn.LayerNorm(512)]
    if head == 'linear':
        layers += [torch.nn.Linear(512,vocab_n)]
    elif head == 'mlp':
        layers += [torch.nn.Linear(512,256),torch.nn.GELU(),torch.nn.Linear(256,vocab_n)]
    else:
        raise ValueError(head)
    return torch.nn.Sequential(*layers)


def read_frames(rows, cache, mode='true', permutation=None):
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence
    seq = []
    for row in rows:
        uid = permutation[row['id']] if mode == 'mismatch' else row['id']
        frame = torch.from_numpy(np.load(cache[uid]['path'],allow_pickle=False).copy()).float()
        assert frame.ndim == 2 and frame.shape[1] == 512 and torch.isfinite(frame).all()
        seq.append(torch.zeros_like(frame) if mode == 'zero' else frame)
    lengths = torch.tensor([len(x) for x in seq],dtype=torch.long)
    return pad_sequence(seq,batch_first=True).cuda(), lengths


def loss_for(model, rows, cache, ids):
    import torch
    frames, lengths = read_frames(rows,cache)
    targets = [torch.tensor([ids[x] for x in r['syllables']],dtype=torch.long) for r in rows]
    target_lengths = torch.tensor([len(x) for x in targets],dtype=torch.long)
    assert all(int(n)>=min_ctc_frames(t.tolist()) for n,t in zip(lengths,targets))
    logits = model(frames)
    log_probs = logits.log_softmax(-1).transpose(0,1).contiguous()
    losses = torch.nn.functional.ctc_loss(log_probs,torch.cat(targets).cuda(),lengths,target_lengths,
                                          blank=0,reduction='none',zero_infinity=False)
    assert torch.isfinite(losses).all()
    return (losses/target_lengths.to(losses.device)).mean(), logits.detach(), lengths


def optimizer(model,c,steps):
    import torch
    opt = torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    warm = c['warmup']
    sch = torch.optim.lr_scheduler.LambdaLR(opt,lambda x: (x+1)/warm if x<warm else
                                           max(0.,(steps-x)/(steps-warm)))
    return opt,sch


def save_checkpoint(model,opt,sch,path,c,epoch,updates):
    import numpy as np
    import torch
    path.mkdir(exist_ok=False)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch,updates=updates),path/'state.pt')
    w.save(path/'metadata.json',dict(complete=True,epoch=epoch,updates=updates,
                                    protocol_sha256=c['protocol_sha256'],state_sha256=w.sha(path/'state.pt')))


def restore(model,opt,sch,path,c):
    import numpy as np
    import torch
    meta=w.read(path/'metadata.json')
    assert meta['complete'] and meta['protocol_sha256']==c['protocol_sha256']
    assert w.sha(path/'state.pt')==meta['state_sha256']
    state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==meta['epoch'] and state['updates']==meta['updates']
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    return meta


def smoke(head):
    import torch
    c,p=configuration(); vocab=w.read(c['vocabulary']); ids={x:i for i,x in enumerate(vocab)}
    cache=w.read(c['cache_manifest']); by_id={r['id']:r for r in p.load(c['train_targets'])}
    rows=[by_id[uid] for uid in c['smoke_ids']]
    with w.gpu_guard(p):
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(6)
        model=make_model(head,len(vocab)).cuda(); model.train()
        opt,sch=optimizer(model,c,math.ceil(c['audit']['train']['n']/c['batch_size'])*c['epochs'])
        before={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        start=time.time(); loss,_,_=loss_for(model,rows,cache,ids); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
        assert torch.isfinite(grad) and grad>0; opt.step(); sch.step()
        assert any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
        model.eval()
        with torch.no_grad():
            expected=model(read_frames(rows,cache)[0]).detach().cpu()
        cp=HERE/head/'smoke_checkpoint'; save_checkpoint(model,opt,sch,cp,c,0,1)
        other=make_model(head,len(vocab)).cuda(); op2,sc2=optimizer(other,c,math.ceil(c['audit']['train']['n']/c['batch_size'])*c['epochs'])
        restore(other,op2,sc2,cp,c); other.eval()
        with torch.no_grad():
            actual=other(read_frames(rows,cache)[0]).detach().cpu()
        assert torch.equal(expected,actual)
        assert sc2.last_epoch==1 and all(int(x['step'])==1 for x in op2.state.values())
        other.train(); op2.zero_grad(set_to_none=True)
        resumed,_,_=loss_for(other,rows,cache,ids); resumed.backward(); op2.step(); sc2.step()
        rec=dict(status='PASS',head=head,loss=float(loss.detach()),resumed_loss=float(resumed.detach()),
                 reload_exact=True,parameters=sum(x.numel() for x in model.parameters()),
                 seconds=time.time()-start,max_gpu_gb=torch.cuda.max_memory_allocated()/1e9,
                 protocol_sha256=c['protocol_sha256'])
        w.save(HERE/head/'smoke.json',rec); print(json.dumps(rec),flush=True)


def epoch_batches(rows,cache,epoch,c):
    # Random minibatches with deterministic local length bucketing reduce padding.
    ordered=list(rows); rng=random.Random(c['seed']+epoch); rng.shuffle(ordered)
    windows=[]
    for i in range(0,len(ordered),c['batch_size']*20):
        window=sorted(ordered[i:i+c['batch_size']*20],key=lambda r:cache[r['id']]['frames'])
        windows.extend(window[j:j+c['batch_size']] for j in range(0,len(window),c['batch_size']))
    rng.shuffle(windows)
    assert sum(map(len,windows))==len(rows)
    return windows


def train(head):
    import torch
    c,p=configuration(); d=HERE/head
    assert w.read(d/'smoke.json')['status']=='PASS'
    assert w.read(d/'smoke.json')['protocol_sha256']==c['protocol_sha256']
    vocab=w.read(c['vocabulary']); ids={x:i for i,x in enumerate(vocab)}
    rows=p.load(c['train_targets']); cache=w.read(c['cache_manifest'])
    batches=math.ceil(len(rows)/c['batch_size']); assert batches==541
    if (d/'selected_checkpoint.json').exists():
        sel=w.read(d/'selected_checkpoint.json'); meta=w.read(Path(sel['checkpoint'])/'metadata.json')
        assert meta['epoch']==10 and meta['updates']==batches*10
        return
    complete=[]
    for q in (d/'checkpoints').glob('epoch_*'):
        if (q/'metadata.json').exists(): complete.append((w.read(q/'metadata.json')['epoch'],q))
    start,cp=max(complete,default=(0,None)); updates=start*batches
    with w.gpu_guard(p):
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(6)
        model=make_model(head,len(vocab)).cuda(); opt,sch=optimizer(model,c,batches*c['epochs'])
        if cp: restore(model,opt,sch,cp,c)
        model.train(); began=time.time(); (d/'checkpoints').mkdir(exist_ok=True)
        with (d/f'train_{time.time_ns()}.jsonl').open('x',encoding='utf-8') as stream:
            for epoch in range(start,c['epochs']):
                seen=0; total=0.
                for batch in epoch_batches(rows,cache,epoch,c):
                    opt.zero_grad(set_to_none=True); loss,_,_=loss_for(model,batch,cache,ids); loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
                    assert torch.isfinite(grad) and grad>0
                    opt.step(); sch.step(); updates+=1; seen+=len(batch); total+=float(loss.detach())*len(batch)
                    rec=dict(status='TRAINING',head=head,epoch=epoch+1,epochs=c['epochs'],updates=updates,
                             seen_this_epoch=seen,train_n=len(rows),loss=float(loss.detach()),
                             seconds=time.time()-began,resume_epoch=start,
                             max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
                    stream.write(json.dumps(rec)+'\n'); stream.flush(); w.save(d/'status.json',rec)
                    if updates%100==0: print(json.dumps(rec),flush=True)
                assert seen==len(rows) and updates==(epoch+1)*batches
                cp=d/'checkpoints'/f'epoch_{epoch+1}_{time.time_ns()}'
                save_checkpoint(model,opt,sch,cp,c,epoch+1,updates)
                w.save(d/f'epoch_{epoch+1}_summary.json',dict(mean_loss=total/seen,updates=updates,examples=seen))
        w.save(d/'selected_checkpoint.json',dict(checkpoint=str(cp),epoch=c['epochs'],reason=c['selection']))
        w.save(d/'status.json',dict(status='TRAINED_NOT_EVALUATED',checkpoint=str(cp)))


def ctc_log_score(log_probs, target, ids):
    import torch
    if any(x not in ids for x in target):
        return None  # Unseen train vocabulary explicitly reported, not silently replaced.
    encoded=[ids[x] for x in target]
    if len(log_probs)<min_ctc_frames(encoded): return None
    if not encoded:
        return float(log_probs[:,0].sum())
    value=torch.nn.functional.ctc_loss(log_probs[:,None,:].contiguous(),
                   torch.tensor(encoded,device=log_probs.device,dtype=torch.long),
                   torch.tensor([len(log_probs)]),torch.tensor([len(encoded)]),
                   blank=0,reduction='sum',zero_infinity=False)
    return -float(value)


def check_rows(rows,expected,mode,c):
    assert len(rows)<=len(expected)
    for r,e in zip(rows,expected):
        assert (r['id'],r['reference'],r['reference_syllables'],r['mode'])==(e['id'],e['reference'],e['syllables'],mode)
        assert r['audio_id']==(c['audio_permutation'][e['id']] if mode=='mismatch' else e['id'])


def evaluate(head):
    import torch
    c,p=configuration(); d=HERE/head; rows=p.load(c['dev_targets'])
    comparisons={r['id']:r for r in p.load(c['comparisons'])}
    vocab=w.read(c['vocabulary']); ids={x:i for i,x in enumerate(vocab)}; cache=w.read(c['cache_manifest'])
    cp=Path(w.read(d/'selected_checkpoint.json')['checkpoint']); meta=w.read(cp/'metadata.json')
    assert meta['epoch']==10 and meta['protocol_sha256']==c['protocol_sha256']
    assert w.sha(cp/'state.pt')==meta['state_sha256']
    dest=d/'dev'; dest.mkdir(exist_ok=True)
    with w.gpu_guard(p):
        torch.set_num_threads(6); model=make_model(head,len(vocab)).cuda()
        state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False); model.load_state_dict(state['model']); model.eval()
        for mode in c['modes']:
            target=dest/(mode+'.predictions.jsonl')
            if target.exists():
                prior_rows=p.load(target); check_rows(prior_rows,rows,mode,c); assert len(prior_rows)==len(rows); continue
            partial=Path(str(target)+'.partial'); prior_rows=p.load(partial) if partial.exists() else []
            check_rows(prior_rows,rows,mode,c)
            with partial.open('a',encoding='utf-8') as stream, torch.no_grad():
                for start in range(len(prior_rows),len(rows),c['batch_size']):
                    batch=rows[start:start+c['batch_size']]
                    frames,lengths=read_frames(batch,cache,mode,c['audio_permutation'])
                    logits=model(frames); assert torch.isfinite(logits).all(); lp=logits.log_softmax(-1)
                    indices=logits.argmax(-1).cpu().tolist()
                    for i,row in enumerate(batch):
                        ref=row['syllables']; pred=greedy(indices[i][:int(lengths[i])],vocab)
                        r=dict(id=row['id'],reference=row['reference'],reference_syllables=ref,predicted_syllables=pred,
                               input=row['input'],head=head,mode=mode,audio_id=c['audio_permutation'][row['id']] if mode=='mismatch' else row['id'])
                        values=lp[i,:int(lengths[i])]; ref_score=ctc_log_score(values,ref,ids)
                        r['reference_ctc_logp']=ref_score; r['baseline_comparisons']={}
                        for key in ('frontend','old_B_text'):
                            bs=comparisons[row['id']][key]
                            score=ctc_log_score(values,bs,ids) if bs is not None else None
                            r['baseline_comparisons'][key]=dict(syllables=bs,eligible=bs is not None,
                                    ctc_logp=score,reference_minus_baseline_logp=(ref_score-score) if ref_score is not None and score is not None else None,
                                    spelling_text=comparisons[row['id']][key+'_text' if key=='frontend' else 'old_B_text_output'])
                        stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                    rec=dict(status='EVALUATING',head=head,mode=mode,done=min(start+len(batch),len(rows)),total=len(rows))
                    w.save(d/'status.json',rec)
                    if rec['done']%256==0 or rec['done']==len(rows): print(json.dumps(rec),flush=True)
            rr=p.load(partial); check_rows(rr,rows,mode,c); assert len(rr)==1334; partial.rename(target)
    w.save(d/'status.json',dict(status='COMPLETE'))


def summarize_rows(rows):
    totals=dict(n=len(rows),syllables=0,errors=0,S=0,D=0,I=0,exact=0,empty=0)
    baseline={key:dict(n=0,syllables=0,baseline_errors=0,probe_errors=0,baseline_correct_positions=0,
                       baseline_correct_positions_changed=0,substitution_positions=0,substitution_positions_recovered=0,
                       missing_positions=0,missing_positions_recovered=0,preference_pairs=0,reference_preferred=0,ties=0,
                       same_phone_pairs=0,same_phone_score_nonzero=0) for key in ('frontend','old_B_text')}
    per=[]
    for r in rows:
        ref=r['reference_syllables']; pred=r['predicted_syllables']; a=align(ref,pred)
        totals['syllables']+=len(ref); totals['errors']+=a['errors']
        for k in ('S','D','I'): totals[k]+=a['edits'][k]
        totals['exact']+=a['errors']==0; totals['empty']+=not pred
        per.append(dict(id=r['id'],syllables=len(ref),errors=a['errors']))
        for key,b in r['baseline_comparisons'].items():
            if not b['eligible']: continue
            z=baseline[key]; ba=align(ref,b['syllables']); z['n']+=1; z['syllables']+=len(ref)
            z['baseline_errors']+=ba['errors']; z['probe_errors']+=a['errors']
            for pos,kind in ba['kinds'].items():
                correct=a['positions'][pos]==ref[pos]
                if kind=='M': z['baseline_correct_positions']+=1; z['baseline_correct_positions_changed']+=not correct
                if kind=='S': z['substitution_positions']+=1; z['substitution_positions_recovered']+=correct
                if kind=='D': z['missing_positions']+=1; z['missing_positions_recovered']+=correct
            gap=b['reference_minus_baseline_logp']
            if gap is not None:
                if ba['errors']==0:
                    z['same_phone_pairs']+=1; z['same_phone_score_nonzero']+=abs(gap)>1e-5
                else:
                    z['preference_pairs']+=1; z['reference_preferred']+=gap>0; z['ties']+=abs(gap)<=1e-5
    assert totals['n']==1334 and totals['syllables']==21102
    totals['syer']=totals['errors']/totals['syllables']
    for z in baseline.values():
        z['baseline_syer']=z['baseline_errors']/z['syllables']; z['probe_syer']=z['probe_errors']/z['syllables']
    return dict(summary=totals,baseline_diagnostics=baseline,per_utterance=per)


def paired_bootstrap(a,b):
    import numpy as np
    assert [r['id'] for r in a]==[r['id'] for r in b]
    den=np.array([r['syllables'] for r in a]); delta=np.array([x['errors']-y['errors'] for x,y in zip(a,b)])
    rng=np.random.default_rng(917); values=[]
    for _ in range(2000):
        ix=rng.integers(len(a),size=len(a)); values.append(100*delta[ix].sum()/den[ix].sum())
    return list(np.percentile(values,[2.5,97.5]))


def report():
    c,p=configuration(); expected=p.load(c['dev_targets']); result=dict(heads={},protocol_sha256=c['protocol_sha256'],limitations=c['limitations'])
    for head in c['heads']:
        metrics={}; hashes={}
        for mode in c['modes']:
            q=HERE/head/'dev'/(mode+'.predictions.jsonl'); rows=p.load(q); check_rows(rows,expected,mode,c)
            metrics[mode]=summarize_rows(rows); hashes[mode]=w.sha(q)
            w.save(q.with_name(mode+'.metrics.json'),metrics[mode])
        result['heads'][head]=dict(summaries={k:v['summary'] for k,v in metrics.items()},
                diagnostics={k:v['baseline_diagnostics'] for k,v in metrics.items()},prediction_hashes=hashes,
                true_minus_control_syer_pp_95ci={mode:paired_bootstrap(metrics['true']['per_utterance'],metrics[mode]['per_utterance']) for mode in ('zero','mismatch')})
    w.save(HERE/'summary.json',result)
    w.save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(result),flush=True)


def pipeline():
    import fcntl
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for head in ('linear','mlp'):
            for task in ('smoke','train','eval'):
                if task=='smoke' and (HERE/head/'smoke.json').exists(): continue
                w.save(HERE/'status.json',dict(status='RUNNING',head=head,phase=task,pid=os.getpid(),updated=time.time()))
                with (HERE/f'{head}_{task}_{time.time_ns()}.log').open('x') as stream:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),task,head],stdout=stream,stderr=subprocess.STDOUT,check=True)
        report()


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','smoke','train','eval','report','pipeline'])
    ap.add_argument('head',nargs='?',choices=['linear','mlp']); args=ap.parse_args()
    try:
        if args.command=='prepare': prepare()
        elif args.command=='smoke': smoke(args.head)
        elif args.command=='train': train(args.head)
        elif args.command=='eval': evaluate(args.head)
        elif args.command=='report': report()
        else: pipeline()
    except BaseException:
        err=dict(status='FAILED',command=vars(args),error=traceback.format_exc(),time=time.time())
        w.save(HERE/f'failure_{time.time_ns()}.json',err); w.save(HERE/'status.json',err); raise
