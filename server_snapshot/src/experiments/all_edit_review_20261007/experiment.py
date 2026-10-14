"""Matched all-edit FULL-only candidate LoRA, using immutable v3 backend."""
import argparse
from collections import Counter
import copy
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
ROOT = Path('/root/autodl-tmp/src/experiments')
PARENT = ROOT / 'evidence_decision_20261005_v3'
from interface import render, swap, preference, label_for, choose, order_bit
spec = importlib.util.spec_from_file_location('immutable_v3_backend', PARENT / 'experiment.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.HERE = HERE
b.render, b.swap, b.preference, b.label_for, b.choose, b.order_bit = render, swap, preference, label_for, choose, order_bit
load, rows, sha, save, write_rows, status = b.load, b.rows, b.sha, b.save, b.write_rows, b.status
e, r, w, p = b.e, b.r, b.w, b.p


def query(manifest, edit, inp, evidence):
    # The route is every input/draft edit, independently of reference/type.
    z = dict(edit)
    z['original_phones'] = z.get('original_phones', e.phones(z['original']))
    z['modified_phones'] = z.get('modified_phones', e.phones(z['modified']))
    return r.make_query(manifest, z, inp, evidence)


def prepare():
    assert not (HERE / 'protocol.json').exists()
    old = load(PARENT / 'protocol.json'); lock = load(PARENT / 'data_lock.json')
    assert lock['protocol_sha256'] == sha(PARENT / 'protocol.json')
    for path, digest in old['sources'].items(): assert sha(path) == digest, path
    for path, digest in lock['hashes'].items(): assert sha(path) == digest, path
    assert (lock['pairs'], lock['draft_failures']) == (5325, 49)
    tr, dv = rows(old['train']), rows(old['dev'])
    drafts = rows(PARENT / 'train_drafts.jsonl'); manifests = rows(PARENT / 'train_edit_manifest.jsonl')
    assert len(tr) == len(drafts) == len(manifests) == 17294 and len(dv) == 1334
    assert [x['id'] for x in tr] == [x['id'] for x in drafts] == [x['id'] for x in manifests]
    assert all(x['split'] == 'train' for x in tr) and all(x['split'] == 'dev' for x in dv)
    for fn in (lambda x: x['id'], lambda x: x['id'].split('W')[0], lambda x: x['wav']):
        assert not {fn(x) for x in tr} & {fn(x) for x in dv}
    types = Counter()
    for row, draft, manifest in zip(tr, drafts, manifests):
        actual = e.candidates(row['id'], row['input']['asr_top1'], draft['prediction'])
        assert actual == manifest, row['id']
        types.update(z['kind'] for z in manifest['edits'])
    assert sum(types.values()) == 18956
    old_dev = rows(old['baseline']); dm = rows(ROOT / 'acoustic_review_20261005/input_edits.jsonl')
    ec = load(ROOT / 'acoustic_review_20261005/evidence_seal.json')
    assert ec['reference_used'] is False
    assert sha(ROOT / 'acoustic_review_20261005/input_edits.jsonl') == ec['manifest_sha256']
    assert [x['id'] for x in old_dev] == [x['id'] for x in dv] == [x['id'] for x in dm]
    controls = {}
    for mode in ('true', 'zero', 'mismatch'):
        path = ROOT / 'acoustic_review_20261005' / (mode + '.evidence.jsonl')
        assert sha(path) == ec['evidence_sha256'][mode]
        controls[mode] = rows(path)
        assert [x['id'] for x in controls[mode]] == [x['id'] for x in dm]
    qs = []
    for i, (row, manifest) in enumerate(zip(old_dev, dm)):
        assert e.candidates(row['id'], row['input']['asr_top1'], row['prediction']) == manifest
        for j, z in enumerate(manifest['edits']):
            scores = {}
            for mode in controls:
                record = controls[mode][i]['edits'][j]
                assert record['edit_id'] == z['edit_id']
                scores[mode] = record['scores']
            qs.append(query(manifest, z, row['input'], scores))
    assert len(qs) == len({q['edit_id'] for q in qs}) == 604
    for name in ('train_drafts.jsonl', 'train_edit_manifest.jsonl'):
        (HERE / name).symlink_to(PARENT / name)
    write_rows(HERE / 'dev_queries.jsonl', qs)
    sources = dict(old['sources'])
    deps = [PARENT / 'experiment.py', PARENT / 'protocol.json', PARENT / 'data_lock.json',
            PARENT / 'train_drafts.jsonl', PARENT / 'train_edit_manifest.jsonl',
            ROOT / 'acoustic_review_20261005/input_edits.jsonl', ROOT / 'acoustic_review_20261005/evidence_seal.json']
    deps += [ROOT / 'acoustic_review_20261005' / (m + '.evidence.jsonl') for m in controls]
    deps += [HERE / n for n in ('experiment.py', 'interface.py', 'test_interface.py', 'audit_prepared.py', 'audit_results.py', 'metric_auditor.py', 'PLAN.md', 'dev_queries.jsonl')]
    for path in deps: sources[str(path)] = sha(path)
    c = copy.deepcopy(old)
    c.update(experiment='All-edit contextual FULL-only A/B review', created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             sources=sources, parent_protocol=str(PARENT / 'protocol.json'), dev_queries=604,
             tasks=['full'], full_weight=1., local_weight=0., expected_train_pairs=18956,
             checkpoint_interval=1000000, revision='All edit types, matched full-context AUDIO/TEXT, no LOCAL objective',
             routing='All input/draft edit blocks; no pronunciation/reference filtering; only original/modified alternatives',
             training_controls='FULL-only on both arms; identical inputs/order/labels except AUDIO true CTC vs TEXT absent',
             train_type_counts=dict(types), no_new_drafts=True, no_new_frontend=True,
             stopping='Any hash, data, finite-gradient, reload, completeness, or GPU-lock failure stops without automatic retries')
    c['limitations'] += ['Both all-edit coverage and FULL-only objective changed vs historical v3; not single-factor historical ablation',
                         'Same-phone evidence cannot establish intended spelling; phonetic dictionary is not gold pronunciation',
                         '49 failed frozen train drafts retained; independent local selections may interact when composed']
    save(HERE / 'protocol.json', c)
    save(HERE / 'candidate_route_seal.json', dict(reference_used=False, train_manifest_sha256=sha(HERE / 'train_edit_manifest.jsonl'),
          train_drafts_sha256=sha(HERE / 'train_drafts.jsonl'), dev_queries_sha256=sha(HERE / 'dev_queries.jsonl'),
          train_pairs=18956, dev_pairs=604, protocol_sha256=sha(HERE / 'protocol.json')))
    status('PREPARED',train_pairs=18956,dev_pairs=604,types=dict(types))


def evidence():
    import numpy as np
    import torch
    from funasr import AutoModel
    c = b.configuration(); tr = rows(c['train']); manifests = rows(HERE / 'train_edit_manifest.jsonl'); cache = load(c['cache_manifest'])
    assert len(tr) == len(manifests) == 17294
    with w.gpu_guard(p), torch.inference_mode():
        torch.set_num_threads(6)
        wrapper = AutoModel(model=str(e.SENSE), device='cuda:0', disable_update=True)
        model = wrapper.model.eval(); tokenizer = wrapper.kwargs['tokenizer']; tokens = load(e.SENSE / 'tokens.json')
        assert model.blank_id == 0 and len(tokens) == model.ctc.ctc_lo.out_features
        groups = {}
        for i, text in enumerate(tokens):
            if len(text) == 1 and '\u4e00' <= text <= '\u9fff':
                ph = e.phones(text)
                if ph is not None: groups.setdefault(ph[0], []).append(i)
        pv = ['<blank>'] + sorted(groups); pi = {s: i for i, s in enumerate(pv)}
        gi = [torch.tensor(groups[s], device='cuda', dtype=torch.long) for s in pv[1:]]
        save(HERE / 'native_interface.json', dict(native_vocab=len(tokens), phone_vocab=pv, mapping=groups, model_sha256=sha(e.SENSE / 'model.pt')))
        output = HERE / 'train_evidence.jsonl'; part = Path(str(output) + '.partial')
        assert not output.exists() and not part.exists()
        began = time.time(); count = 0
        with part.open('x', encoding='utf-8') as f:
            for i, (row, manifest) in enumerate(zip(tr, manifests)):
                assert row['id'] == manifest['id']
                result = dict(id=row['id'], queries=[])
                if manifest['edits']:
                    info = cache[row['id']]; assert sha(info['path']) == info['sha256']
                    frames = torch.from_numpy(np.load(info['path'], allow_pickle=False).copy()).float().cuda()
                    assert tuple(frames.shape) == (info['frames'],512) and torch.isfinite(frames).all()
                    lp = model.ctc.log_softmax(frames[None])[0]; assert torch.isfinite(lp).all()
                    plp = torch.stack([lp[:,0]] + [torch.logsumexp(lp[:,g],dim=-1) for g in gi],dim=-1)
                    plp -= torch.logsumexp(plp,dim=-1,keepdim=True)
                    memo = {}
                    def calc(text, ph):
                        key = (text, tuple(ph) if ph is not None else None)
                        if key not in memo:
                            ids = [int(x) for x in tokenizer.encode(text)]
                            bad = any(x == 0 or 1 <= x < 16 or x >= 25000 for x in ids)
                            native = dict(score=None,reason='unknown_or_special_native_token') if bad else e.ctc_score(lp,ids)
                            phone = dict(score=None,reason='non_Han_or_unknown_pronunciation') if ph is None else e.ctc_score(plp,[pi.get(x) for x in ph])
                            memo[key] = dict(native=native,phonetic_marginal=phone)
                        return memo[key]
                    for z in manifest['edits']:
                        aa = calc(z['candidate_original_context'],z['original_context_phones'])
                        bb = calc(z['candidate_modified_context'],z['modified_context_phones']); scores = {}
                        for method in ('native','phonetic_marginal'):
                            a,v = aa[method],bb[method]
                            scores[method] = dict(original=a,modified=v,modified_minus_original=v['score']-a['score'] if a['score'] is not None and v['score'] is not None else None)
                        result['queries'].append(query(manifest,z,row['input'],{'true':scores})); count += 1
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
                if (i+1) % 100 == 0 or i+1 == len(tr): status('TRAIN_EVIDENCE',done=i+1,total=len(tr),pairs=count,seconds=time.time()-began)
        assert count == 18956 and [x['id'] for x in rows(part)] == [x['id'] for x in tr]
        part.rename(output)
    save(HERE / 'train_evidence_seal.json', dict(protocol_sha256=c['protocol_sha256'],reference_used=False,
          train_drafts_sha256=sha(HERE/'train_drafts.jsonl'),manifest_sha256=sha(HERE/'train_edit_manifest.jsonl'),evidence_sha256=sha(output)))
    status('TRAIN_EVIDENCE_COMPLETE',pairs=count)


def labels():
    import torch
    from transformers import AutoTokenizer
    c = b.configuration(); seal = load(HERE / 'train_evidence_seal.json')
    assert seal['protocol_sha256'] == c['protocol_sha256'] and not seal['reference_used']
    assert seal['evidence_sha256'] == sha(HERE / 'train_evidence.jsonl')
    tr = {x['id']:x for x in rows(c['train'])}; qs = [q for x in rows(HERE/'train_evidence.jsonl') for q in x['queries']]
    assert len(qs) == len({q['edit_id'] for q in qs}) == 18956
    labs = [preference(q,tr[q['id']]['reference'],e.norm,lambda a,v:e.align(a,v)[0],'train') for q in qs]
    write_rows(HERE/'train_queries.jsonl',qs); write_rows(HERE/'train_labels.jsonl',labs)
    tok = AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
    choice = [tok(x,add_special_tokens=False)['input_ids'] for x in ('A','B')]
    assert all(len(x)==1 for x in choice) and choice[0]!=choice[1]
    bank = {}; longest = (0,None)
    def encode(q,mode):
        nonlocal longest
        key = (q['edit_id'],q['A']['role'],mode,'full')
        if key in bank:return
        prompt = tok.apply_chat_template(render(q,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids = tok(prompt,add_special_tokens=False,return_tensors='pt').input_ids
        assert ids.shape[0]==1 and 0<ids.shape[1]<=c['max_sequence'],q['edit_id']
        bank[key]=ids
        if q['id'] in tr and ids.shape[1]>longest[0]:longest=(ids.shape[1],q['edit_id'])
    for i,q in enumerate(qs):
        for mode in ('none','true'):encode(swap(q) if order_bit(q['edit_id']) else q,mode)
        if (i+1)%1000==0:status('PRETOKENIZING',done=i+1,total=len(qs))
    smoke_ids=list(dict.fromkeys([qs[0]['edit_id'],next(q['edit_id'] for q,l in zip(qs,labs) if not l['neutral']),
                                 next(q['edit_id'] for q,l in zip(qs,labs) if l['neutral']),longest[1]]))
    by={q['edit_id']:q for q in qs}
    for uid in smoke_ids:
        for mode in ('none','true'):encode(by[uid],mode)
    for q in rows(HERE/'dev_queries.jsonl'):
        for mode in c['evaluation_modes']:encode(q,mode)
        encode(swap(q),'true')
    torch.save(bank,HERE/'token_bank.pt')
    save(HERE/'token_bank_metadata.json',dict(reference_used=False,entries=len(bank),max_tokens=max(v.shape[1] for v in bank.values()),token_bank_sha256=sha(HERE/'token_bank.pt')))
    order=[q['edit_id'] for q in qs];random.Random(917).shuffle(order)
    files=('train_drafts.jsonl','train_edit_manifest.jsonl','train_evidence.jsonl','train_evidence_seal.json','train_queries.jsonl',
           'train_labels.jsonl','dev_queries.jsonl','candidate_route_seal.json','token_bank.pt','token_bank_metadata.json')
    d=dict(protocol_sha256=c['protocol_sha256'],hashes={str(HERE/n):sha(HERE/n) for n in files},pairs=len(qs),updates=math.ceil(len(qs)/8),
           choice_token_ids=[x[0] for x in choice],train_order=order,smoke_ids=smoke_ids,train_utterances=17294,
           utterances_with_pairs=len({q['id'] for q in qs}),neutral_pairs=sum(x['neutral'] for x in labs),
           prefer_A=sum(x['prob_A']==1 for x in labs),prefer_B=sum(x['prob_A']==0 for x in labs),
           max_prompt_tokens=max(v.shape[1] for v in bank.values()),draft_failures=sum(not x['parsed'] for x in rows(HERE/'train_drafts.jsonl')))
    assert d['draft_failures']==49 and d['updates']==2370
    save(HERE/'data_lock.json',d);status('TRAIN_DATA_LOCKED',pairs=d['pairs'],updates=d['updates'],max_tokens=d['max_prompt_tokens'])


def update(rt,chunk,opt,sch,arm):
    import torch
    opt.zero_grad(set_to_none=True);values=[]
    for q,label in chunk:
        reverse=bool(order_bit(q['edit_id']));qq=swap(q) if reverse else q;y=label_for(q,label,reverse)
        target=torch.tensor([y,1-y],device='cuda',dtype=torch.float32)
        logits=rt.logits(qq,'true' if arm=='AUDIO' else 'none','full')
        loss=-(target*torch.log_softmax(logits,dim=0)).sum()
        (loss/len(chunk)).backward();values.append(loss.detach())
    logged=torch.stack(values);assert torch.isfinite(logged).all()
    assert all(v.grad is None for v in rt.model.parameters() if not v.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(rt.params,rt.c['clip']);assert torch.isfinite(norm) and norm>0
    opt.step();sch.step()
    return dict(loss=sum(logged.cpu().tolist())/len(chunk),grad_norm=float(norm),lr=sch.get_last_lr()[0])


b.update=update


def evaluate(arm):
    import torch
    c,d=b.data_config();directory=HERE/arm;directory.mkdir(exist_ok=True)
    cp=None if arm=='UNADAPTED' else directory/'final_checkpoint'
    if cp is not None:b.verify_checkpoint(cp,c,arm)
    qs=rows(HERE/'dev_queries.jsonl');assert len(qs)==604
    with w.gpu_guard(p):
        rt=b.Runtime(c,d,False,cp);began=time.time();hashes={}
        for mode in c['evaluation_modes']+['true_reversed']:
            path=directory/(mode+'.decisions.jsonl');part=Path(str(path)+'.partial');assert not path.exists() and not part.exists()
            with part.open('x',encoding='utf-8') as f,torch.inference_mode():
                for i,q in enumerate(qs):
                    qq=swap(q) if mode=='true_reversed' else q;condition='true' if mode=='true_reversed' else mode
                    ll=rt.logits(qq,condition).cpu().tolist();key=choose(*ll)
                    record=dict(id=q['id'],edit_id=q['edit_id'],arm=arm,mode=mode,logits=ll,selected_key=key,selected_role=qq[key]['role'],
                                candidate_roles={k:qq[k]['role'] for k in ('A','B')},batch_size=1,padding_tokens=0)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                    if (i+1)%50==0 or i+1==len(qs):status('DECISION_VALIDATION',arm=arm,mode=mode,done=i+1,total=len(qs),seconds=time.time()-began)
            assert [x['edit_id'] for x in rows(part)]==[q['edit_id'] for q in qs];part.rename(path);hashes[mode]=sha(path)
        save(directory/'decision_seal.json',dict(protocol_sha256=c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json'),
             hashes=hashes,reference_used=False,checkpoint=str(cp or c['checkpoint'])))
    status('DECISIONS_SEALED',arm=arm)


def report():
    # Only after all three arms have sealed predictions may dev reference diagnosis be read.
    c,d=b.data_config();qs=rows(HERE/'dev_queries.jsonl')
    for arm in ('UNADAPTED','AUDIO','TEXT'):
        s=load(HERE/arm/'decision_seal.json');assert s['protocol_sha256']==c['protocol_sha256'] and not s['reference_used']
        for mode,digest in s['hashes'].items():assert sha(HERE/arm/(mode+'.decisions.jsonl'))==digest
    old=rows(c['baseline']);baseline=p.metrics(old,protected=False)
    assert (baseline['summary']['errors'],baseline['summary']['hits'])==(622,469)
    old_by={x['id']:x for x in old};man={x['id']:x for x in rows(ROOT/'acoustic_review_20261005/input_edits.jsonl')}
    kind={z['edit_id']:z['kind'] for x in man.values() for z in x['edits']}
    labels={}
    for q in qs:
        ref=e.norm(old_by[q['id']]['reference'])
        err={q[k]['role']:e.align(ref,e.norm(q[k]['sentence']))[0] for k in ('A','B')}
        labels[q['edit_id']]='harmful' if err['original']<err['modified'] else 'helpful' if err['original']>err['modified'] else 'neutral'
    write_rows(HERE/'dev_posthoc_labels.jsonl',[dict(edit_id=q['edit_id'],kind=kind[q['edit_id']],diagnosis=labels[q['edit_id']]) for q in qs])
    metrics={};cells={};type_cells={};agreement={};hashes={}
    for arm in ('UNADAPTED','AUDIO','TEXT'):
        directory=HERE/arm;normal=rows(directory/'true.decisions.jsonl');rev=rows(directory/'true_reversed.decisions.jsonl')
        assert [x['edit_id'] for x in rev]==[q['edit_id'] for q in qs]
        agreement[arm]=sum(a['selected_role']==v['selected_role'] for a,v in zip(normal,rev))
        for mode in c['evaluation_modes']:
            ds=rows(directory/(mode+'.decisions.jsonl'));assert [x['edit_id'] for x in ds]==[q['edit_id'] for q in qs]
            selected={};cc={k:dict(n=0,original=0,modified=0) for k in ('harmful','helpful','neutral')};tc={}
            for q,x in zip(qs,ds):
                assert choose(*x['logits'])==x['selected_key'];selected.setdefault(q['id'],[]).append((q,x['selected_role']))
                cat=labels[q['edit_id']];z=cc[cat];z['n']+=1;z[x['selected_role']]+=1
                typed=tc.setdefault(kind[q['edit_id']],{k:dict(n=0,original=0,modified=0) for k in cc})
                typed[cat]['n']+=1;typed[cat][x['selected_role']]+=1
            out=[]
            for row in old:
                text=e.norm(row['prediction']) if row['id'] in selected else row['prediction']
                for q,role in reversed(selected.get(row['id'],[])):
                    lo,hi=q['output_span'];orig=next(q[k] for k in ('A','B') if q[k]['role']=='original');mod=next(q[k] for k in ('A','B') if q[k]['role']=='modified')
                    assert text[lo:hi]==mod['text']
                    if role=='original':text=text[:lo]+orig['text']+text[hi:]
                out.append(dict(id=row['id'],input=row['input'],reference=row['reference'],prediction=text,parsed=True,condition=arm+'__'+mode))
            assert [x['id'] for x in out]==[x['id'] for x in old]
            write_rows(directory/(mode+'.predictions.jsonl'),out);mm=p.metrics(out,protected=False);mm['diagnostics']=r.diagnose(out,p.evaluator().norm)
            assert (mm['summary']['n'],mm['summary']['chars'],mm['summary']['mentions'])==(1334,21102,599)
            save(directory/(mode+'.metrics.json'),mm);key=arm+'__'+mode;metrics[key]=mm;cells[key]=cc;type_cells[key]=tc;hashes[key]=sha(directory/(mode+'.predictions.jsonl'))
    contrasts={}
    for arm in ('AUDIO','TEXT'):
        a=arm+'__true';contrasts[a+'_vs_old']=p.bootstrap(metrics[a]['per_utterance'],baseline['per_utterance'])
        for mode in ('none','zero','mismatch'):
            v=arm+'__'+mode;contrasts[a+'_vs_'+v]=p.bootstrap(metrics[a]['per_utterance'],metrics[v]['per_utterance'])
    contrasts['TEXT__none_vs_old']=p.bootstrap(metrics['TEXT__none']['per_utterance'],baseline['per_utterance'])
    for v in ('TEXT__none','TEXT__true','UNADAPTED__true','UNADAPTED__none'):
        contrasts['AUDIO__true_vs_'+v]=p.bootstrap(metrics['AUDIO__true']['per_utterance'],metrics[v]['per_utterance'])
    save(HERE/'summary.json',dict(status='COMPLETE',protocol_sha256=c['protocol_sha256'],data_lock_sha256=sha(HERE/'data_lock.json'),
         baseline=baseline['summary'],summaries={k:v['summary'] for k,v in metrics.items()},diagnostics={k:v['diagnostics'] for k,v in metrics.items()},
         decisions=cells,decisions_by_type=type_cells,paired95ci=contrasts,true_order_agreement_of604=agreement,prediction_hashes=hashes,limitations=c['limitations']))
    status('ALL_DONE')


def pipeline():
    jobs=[('prepare',None),('evidence',None),('labels',None),('audit-prepared',None),('smoke','AUDIO'),('smoke-reload','AUDIO'),
          ('smoke','TEXT'),('smoke-reload','TEXT'),('train','AUDIO'),('train','TEXT'),('evaluate','UNADAPTED'),('evaluate','AUDIO'),('evaluate','TEXT'),('report',None),('audit-results',None)]
    for cmd,arm in jobs:
        status('PIPELINE_STAGE_START',command=cmd,arm=arm)
        with (HERE/(cmd+('_'+arm if arm else '')+'.log')).open('x',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(HERE/'experiment.py'),cmd]+([arm] if arm else []),stdout=f,stderr=subprocess.STDOUT,check=True,cwd=HERE)
    status('ALL_DONE_AUDITED')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pipeline','prepare','evidence','labels','audit-prepared','smoke','smoke-reload','train','evaluate','report','audit-results']);ap.add_argument('arm',nargs='?',choices=['AUDIO','TEXT','UNADAPTED']);args=ap.parse_args()
    try:
        if args.command=='smoke-reload':b.smoke(args.arm,True)
        elif args.command in ('smoke','train'):getattr(b,args.command)(args.arm)
        elif args.command=='evaluate':evaluate(args.arm)
        elif args.command in ('audit-prepared','audit-results'):
            subprocess.run([sys.executable,str(HERE/('audit_prepared.py' if args.command=='audit-prepared' else 'audit_results.py')),str(HERE)],check=True)
        else:globals()[args.command]()
    except BaseException:
        failure=dict(error=traceback.format_exc(),when=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=args.command,arm=args.arm)
        save(HERE/('failure_'+str(time.time_ns())+'.json'),failure);status('FAILED',**failure);raise
