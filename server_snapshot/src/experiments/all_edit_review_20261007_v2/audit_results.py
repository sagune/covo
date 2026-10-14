"""Independent full reconstruction, metric and paired-CI audit after decision seals."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
from audit_prepared import rows,load,sha,norm,align,distance,ROOT

spec=importlib.util.spec_from_file_location('immutable_metric_auditor',Path(__file__).with_name('metric_auditor.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def audit(here):
    c,d,s=(load(here/n) for n in ('protocol.json','data_lock.json','summary.json'))
    ph,dh=sha(here/'protocol.json'),sha(here/'data_lock.json')
    assert s['status']=='COMPLETE' and d['protocol_sha256']==s['protocol_sha256']==ph and s['data_lock_sha256']==dh
    for path,digest in {**c['sources'],**d['hashes']}.items():assert sha(path)==digest,path
    old=rows(c['baseline']);qs=rows(here/'dev_queries.jsonl');ids=[r['id'] for r in old];qids=[q['edit_id'] for q in qs]
    assert len(ids)==len(set(ids))==1334 and len(qids)==len(set(qids))==604
    annotations={}
    for line in Path('/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev/aligned.txt').read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('\t')
        if len(fields)>=2:annotations.setdefault(fields[1].strip(),[]).append(norm(fields[0]))
    base,base_diag,baseper=m.score(old,annotations);assert base==s['baseline']
    assert (base['errors'],base['hits'],base['chars'],base['mentions'])==(622,469,21102,599)
    post={r['edit_id']:r for r in rows(here/'dev_posthoc_labels.jsonl')};oldby={x['id']:x for x in old}
    for q in qs:
        er={q[k]['role']:distance(norm(oldby[q['id']]['reference']),norm(q[k]['sentence'])) for k in ('A','B')}
        cat='harmful' if er['original']<er['modified'] else 'helpful' if er['original']>er['modified'] else 'neutral'
        assert post[q['edit_id']]['diagnosis']==cat
    allper={'old':baseper};counts={};changed={};choice_changes={}
    for arm in ('UNADAPTED','AUDIO','TEXT'):
        folder=here/arm;seal=load(folder/'decision_seal.json')
        assert seal['protocol_sha256']==ph and seal['data_lock_sha256']==dh and not seal['reference_used']
        assert set(seal['hashes'])=={'none','true','zero','mismatch','true_reversed'}
        raw={}
        for mode,digest in seal['hashes'].items():
            path=folder/(mode+'.decisions.jsonl');assert sha(path)==digest
            rr=rows(path);assert [x['edit_id'] for x in rr]==qids
            for q,x in zip(qs,rr):
                assert x['id']==q['id'] and x['arm']==arm and x['mode']==mode and x['batch_size']==1 and x['padding_tokens']==0
                a,v=x['logits'];assert math.isfinite(a) and math.isfinite(v)
                key='A' if a>=v else 'B';assert x['selected_key']==key
                qq=dict(q,A=q['B'],B=q['A']) if mode=='true_reversed' else q
                assert x['selected_role']==qq[key]['role']
            raw[mode]=rr
        assert sum(x['selected_role']==y['selected_role'] for x,y in zip(raw['true'],raw['true_reversed']))==s['true_order_agreement_of604'][arm]
        choice_changes[arm]={mode:sum(x['selected_role']!=y['selected_role'] for x,y in zip(raw['true'],raw[mode])) for mode in ('none','zero','mismatch')}
        if arm!='UNADAPTED':
            meta=load(folder/'final_checkpoint/metadata.json')
            assert meta['complete'] and meta['step']==2370 and meta['examples']==18956 and meta['protocol_sha256']==ph and meta['data_lock_sha256']==dh
            assert load(folder/'smoke_reload.json')['exact_resumed_adapter']
            assert meta['initial_adapter_hash']==sha(Path(c['checkpoint'])/'adapter/adapter_model.safetensors')
            for name,digest in meta['hashes'].items():assert sha(folder/'final_checkpoint'/name)==digest
            stress=load(here/('memory_stress_'+arm+'.json'))
            assert stress['status']=='PASS' and stress['discarded_weights'] is True and stress['no_gpu_token_cache'] is True
            assert stress['conservative_headroom_bytes']>=4*1024**3 and stress['protocol_sha256']==ph
            assert [x['edit_id'] for x in rows(folder/'train_order.jsonl')]==d['train_order']
            effective=[]
            for logfile in sorted(folder.glob('training_attempt_*.jsonl')):
                attempt=load(folder/('attempt_'+logfile.stem.split('_')[-1]+'.json'))
                assert attempt['protocol_sha256']==ph and attempt['data_lock_sha256']==dh
                records=rows(logfile)
                assert [x['step'] for x in records]==list(range(attempt['starting_completed_step']+1,attempt['starting_completed_step']+1+len(records)))
                assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in records)
                effective=[x for x in effective if x['step']<=attempt['starting_completed_step']]+records
            assert effective==rows(folder/'training.jsonl')
            assert [x['step'] for x in effective]==list(range(1,2371))
            assert [x['pairs'] for x in effective]==[min(i*8,18956) for i in range(1,2371)]
            for step in range(128,2370,128):
                cp=folder/('checkpoint_'+str(step));record=load(cp/'metadata.json')
                assert record['complete'] and record['step']==step and record['examples']==step*8
                assert record['protocol_sha256']==ph and record['data_lock_sha256']==dh
                for name,digest in record['hashes'].items():assert sha(cp/name)==digest
        for mode in c['evaluation_modes']:
            key=arm+'__'+mode;path=folder/(mode+'.predictions.jsonl');assert sha(path)==s['prediction_hashes'][key]
            pred=rows(path);assert [x['id'] for x in pred]==ids
            choices={};cc={k:dict(n=0,original=0,modified=0) for k in ('harmful','helpful','neutral')};tc={}
            for q,x in zip(qs,raw[mode]):
                choices.setdefault(q['id'],[]).append((q,x['selected_role']))
                lab=post[q['edit_id']];cat=lab['diagnosis'];cc[cat]['n']+=1;cc[cat][x['selected_role']]+=1
                obj=tc.setdefault(lab['kind'],{k:dict(n=0,original=0,modified=0) for k in cc});obj[cat]['n']+=1;obj[cat][x['selected_role']]+=1
            assert cc==s['decisions'][key] and tc==s['decisions_by_type'][key]
            for row,actual in zip(old,pred):
                if row['id'] not in choices:expected=row['prediction']
                else:
                    draft=norm(row['prediction']);parts=[];cursor=0
                    for q,role in choices[row['id']]:
                        lo,hi=q['output_span'];assert cursor<=lo
                        mod=next(q[k]['text'] for k in ('A','B') if q[k]['role']=='modified');assert draft[lo:hi]==mod
                        picked=next(q[k]['text'] for k in ('A','B') if q[k]['role']==role)
                        parts += [draft[cursor:lo],picked];cursor=hi
                    parts.append(draft[cursor:]);expected=''.join(parts)
                assert actual['prediction']==expected and actual['parsed'] is True
            total,diag,per=m.score(pred,annotations)
            assert (total['n'],total['chars'],total['mentions'])==(1334,21102,599)
            assert total==s['summaries'][key] and per==load(folder/(mode+'.metrics.json'))['per_utterance']
            for name,val in diag.items():assert val==s['diagnostics'][key][name]
            allper[key]=per;counts[key]=dict(total,**diag)
            changed[key]=dict(fixed_errors=sum(max(a['errors']-v['errors'],0) for a,v in zip(baseper,per)),
                              new_errors=sum(max(v['errors']-a['errors'],0) for a,v in zip(baseper,per)),
                              fixed_sentences=sum(v['errors']<a['errors'] for a,v in zip(baseper,per)),
                              worsened_sentences=sum(v['errors']>a['errors'] for a,v in zip(baseper,per)))
    for name,expected in s['paired95ci'].items():
        a,v=name.split('_vs_');actual=m.ci(allper[a],allper[v])
        for k,val in actual.items():assert all(abs(x-y)<1e-12 for x,y in zip(val,expected[k])),(name,k)
    out=dict(status='PASS',groups=12,queries_per_group=604,predictions_per_group=1334,reference_chars=21102,hotword_mentions=599,
             protocol_sha256=ph,data_lock_sha256=dh,summary_sha256=sha(here/'summary.json'),
             all_prediction_reconstruction_verified=True,all_metrics_SDI_CI_verified=True,counts=counts,vs_old=changed,
             evidence_selection_changes_of604=choice_changes,limitations=c['limitations'])
    with (here/'independent_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':audit(Path(sys.argv[1]))


