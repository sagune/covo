"""Independent post-seal score/choice/output/metric audit, never changes predictions."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'acoustic-review-20261005'))
from review import norm, align


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def bootstrap(a,b):
    assert [x['id'] for x in a]==[x['id'] for x in b]
    rng=np.random.default_rng(917);cer=[];recall=[]
    er=np.array([x['errors']-y['errors'] for x,y in zip(a,b)])
    hi=np.array([x['hits']-y['hits'] for x,y in zip(a,b)])
    ch=np.array([x['chars'] for x in a]);me=np.array([x['mentions'] for x in a])
    for _ in range(2000):
        idx=rng.integers(0,len(a),len(a));cer.append(100*er[idx].sum()/ch[idx].sum())
        if me[idx].sum():recall.append(100*hi[idx].sum()/me[idx].sum())
    return dict(cer_delta_pp_95ci=np.quantile(cer,[.025,.975]).tolist(),recall_delta_pp_95ci=np.quantile(recall,[.025,.975]).tolist())


def main():
    root=Path(sys.argv[1]);p=load(root/'protocol.json');s=load(root/'summary.json');seal=load(root/'score_seal.json')
    assert not seal['reference_used']
    assert sha(root/'protocol.json')==seal['protocol_sha256']==s['protocol_sha256']
    runtime=load(root/'runtime_audit.json')
    assert runtime['status']=='PASS' and runtime['single_no_padding_repeat_exact']
    assert runtime['first8_scores']==runtime['first8_repeat_scores']
    assert p['no_padding'] and p['batch_sequences']==1
    qs=rows(root/'queries.jsonl');assert len(qs)==len({q['edit_id'] for q in qs})==190
    assert len({q['id'] for q in qs})==180
    for q in qs:
        assert set(q)=={'id','edit_id','output_span','draft','input_evidence','A','B','evidence'}
        assert set(q['input_evidence'])=={'asr_top1','nbest','hotwords'}
        bit=int(hashlib.sha256(('917|candidate_order|'+q['edit_id']).encode()).hexdigest()[:16],16)%2
        assert q['A']['role']==('modified' if bit else 'original')
    baseline=rows(root/'B/dev/text.predictions.jsonl');ids=[b['id'] for b in baseline]
    assert len(ids)==len(set(ids))==1334
    annotation={}
    for line in (root/'aligned.txt').read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('\t')
        if len(fields)>=2:annotation.setdefault(fields[1].strip(),[]).append(norm(fields[0]))
    counts={};all_scores={};metrics={};source_checks={}
    for path,expected in p['sources'].items():
        if '/candidate_likelihood_single_20261005/' in path:local=root/path.split('/candidate_likelihood_single_20261005/',1)[1]
        elif '/candidate_likelihood_20261005/' in path:local=root.parent.parent/'candidate-likelihood-20261005/failed-audited'/path.split('/candidate_likelihood_20261005/',1)[1]
        elif '/acoustic_rereview_20261005/' in path:local=root.parent.parent/'acoustic-rereview-20261005/audited'/path.split('/acoustic_rereview_20261005/',1)[1]
        elif '/acoustic_review_20261005/' in path:local=root.parent.parent/'acoustic-review-20261005/audited'/path.split('/acoustic_review_20261005/',1)[1]
        elif '/joint_lora_20260923/B/' in path:local=root/'B'/path.split('/joint_lora_20260923/B/',1)[1]
        elif path.endswith('/aligned.txt'):local=root/'aligned.txt'
        else:continue
        if local.exists():assert sha(local)==expected,path;source_checks[path]=expected
    for cc in p['conditions']:
        name=cc['name'];score=rows(root/(name+'.scores.jsonl'));pred=rows(root/(name+'.predictions.jsonl'))
        assert len(score)==190 and [z['edit_id'] for z in score]==[q['edit_id'] for q in qs]
        assert len(pred)==1334 and [z['id'] for z in pred]==ids
        assert sha(root/(name+'.scores.jsonl'))==seal['hashes'][name]==s['score_hashes'][name]
        assert sha(root/(name+'.predictions.jsonl'))==s['prediction_hashes'][name]
        all_scores[name]=score;pending={}
        for q,x in zip(qs,score):
            assert x['id']==q['id'] and x['condition']==name
            cand={k:q[k] for k in ('A','B')}
            if cc['order']=='reversed_order':cand['A'],cand['B']=cand['B'],cand['A']
            assert {k:cand[k]['role'] for k in ('A','B')}==x['candidate_roles']
            for k in ('A','B'):
                a=x['candidates'][k]
                assert a['model_batch_size']==1 and a['padding_tokens']==0
                assert a['sequence_tokens']==a['prompt_tokens']+a['target_tokens']
                assert a['target_tokens']==len(a['target_ids'])==len(a['token_logp'])>0
                assert all(math.isfinite(v) for v in a['token_logp'])
                assert math.isclose(math.fsum(a['token_logp']),a['logp'],abs_tol=1e-10)
                assert math.isclose(a['logp']/a['target_tokens'],a['mean_logp'],abs_tol=1e-10)
            a=x['candidates']['A']['logp'];b=x['candidates']['B']['logp']
            key='A' if a>=b else 'B'
            assert key==x['selected_key'] and cand[key]['role']==x['selected_role']
            assert x['exact_tie']==(a==b) and math.isclose(x['logp_A_minus_B'],a-b,abs_tol=1e-10)
            pending.setdefault(q['id'],[]).append((q,x['selected_role']))
        total=dict(n=1334,chars=0,errors=0,input_errors=0,hits=0,mentions=0,exact=0,parse_failures=0,
            S=0,D=0,I=0,input_correct_sentences=0,correct_input_broken_sentences=0,errors_on_correct_inputs=0)
        for b,x in zip(baseline,pred):
            assert (b['id'],b['reference'],b['input'])==(x['id'],x['reference'],x['input'])
            ps=pending.get(b['id'],[]);expected=norm(b['prediction']) if ps else b['prediction']
            for q,role in reversed(ps):
                lo,hi=q['output_span'];original=next(q[k] for k in ('A','B') if q[k]['role']=='original')
                modified=next(q[k] for k in ('A','B') if q[k]['role']=='modified')
                assert expected[lo:hi]==modified['text']
                if role=='original':expected=expected[:lo]+original['text']+expected[hi:]
            assert x['prediction']==expected and x['parsed']
            ref=norm(x['reference']);text=norm(x['prediction']);inp=norm(x['input']['asr_top1'])
            er,ops=align(ref,text);terms=annotation.get(x['id'],[])
            total['chars']+=len(ref);total['errors']+=er;total['input_errors']+=align(ref,inp)[0]
            total['mentions']+=len(terms);total['hits']+=sum(t in text for t in terms);total['exact']+=er==0
            for op in ops:
                if op[0] in ('S','D','I'):total[op[0]]+=1
            if ref==inp:
                total['input_correct_sentences']+=1;total['correct_input_broken_sentences']+=text!=ref;total['errors_on_correct_inputs']+=er
        for k in ('n','chars','errors','input_errors','hits','mentions','exact','parse_failures'):assert total[k]==s['summaries'][name][k],(name,k)
        for k in ('S','D','I','input_correct_sentences','correct_input_broken_sentences','errors_on_correct_inputs'):assert total[k]==s['diagnostics'][name][k],(name,k)
        assert (total['n'],total['chars'],total['mentions'])==(1334,21102,599)
        counts[name]=total;metrics[name]=load(root/(name+'.metrics.json'))['per_utterance']
    base_metrics=load(root/'B/dev/text.metrics.json')['per_utterance']
    for name,mm in metrics.items():
        ci=bootstrap(mm,base_metrics)
        for k,v in ci.items():assert np.allclose(v,s['paired95ci_vs_old'][name][k],atol=1e-12)
    for order in ('original_order','reversed_order'):
        for mode in ('none','zero','mismatch'):
            ci=bootstrap(metrics['true__'+order],metrics[mode+'__'+order])
            for k,v in ci.items():assert np.allclose(v,s['true_minus_controls95ci'][order][mode][k],atol=1e-12)
    for mode in ('none','true','zero','mismatch'):
        same=sum(a['selected_role']==b['selected_role'] for a,b in zip(all_scores[mode+'__original_order'],all_scores[mode+'__reversed_order']))
        assert same==s['order_role_agreement_of190'][mode]
    audit=dict(status='PASS',groups=8,scores_per_group=190,predictions_per_group=1334,counts=counts,
        raw_token_logp_sums_verified=True,candidate_choices_verified=True,independent_output_reconstruction=True,
        full_metrics_and_paired95ci_verified=True,source_hash_checks=source_checks)
    (root/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
