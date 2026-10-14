"""CPU-only retrospective audit; never called by training or decoding."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import numpy as np
from local_inputs import norm, pronunciation, label_slots, aligned_characters

H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_trial_posthoc',H/'experiment.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
c,p=e.configuration()
rows=p.load(c['dev']); assert len(rows)==1334
slots={x['id']:x for x in p.load(c['input_slots'])}
summary=e.w.read(H/'summary.json')
assert e.w.read(H/'status.json')['status']=='ALL_DONE'
assert summary['protocol_sha256']==c['protocol_sha256']
initial=e.w.read(H/'initialization.json')['tensor_sha256']
for arm in e.ARMS:
    q=Path(e.w.read(H/arm/'selected_checkpoint.json')['checkpoint'])
    m=e.check_checkpoint(q,c,arm)
    assert (m['step'],m['examples'],m['initial_tensor_sha256'])==(2162,17294,initial)
    assert e.w.read(H/arm/'smoke.json')['initial_tensor_sha256']==initial

def alignment(a,b):
    n,m=len(a),len(b);d=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+(a[i-1]!=b[j-1]),d[i-1][j]+1,d[i][j-1]+1)
    ops=[];i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:ops.append(('S',i-1,j-1))
            i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:ops.append(('D',i-1,None));i-=1
        else:ops.append(('I',None,j-1));j-=1
    return list(reversed(ops))

anns=collections.defaultdict(list)
for line in p.ALIGNED.read_text(encoding='utf-8-sig').splitlines():
    f=line.split('\t')
    if len(f)>=2:anns[f[1].strip()].append(norm(f[0]))
preds={'old_text':p.load(e.reader.old.HERE/'B/dev/text.predictions.jsonl')}
for arm in e.ARMS:
    for mode in e.MODES:preds[f'{arm}_{mode}']=p.load(H/arm/'dev'/f'{mode}.predictions.jsonl')
audits={};metrics={};per={};classes={};pair_vectors={};details={}
for name,rr in preds.items():
    mode='text' if name=='old_text' else name.split('_',1)[1]
    e.reader.old.check_predictions(rr,rows,mode,c['audio_permutation'])
    assert len(rr)==1334
    file=e.reader.old.HERE/'B/dev/text.predictions.jsonl' if name=='old_text' else H/name.split('_',1)[0]/'dev'/f'{mode}.predictions.jsonl'
    assert e.w.sha(file)==summary['prediction_hashes'][name]
    m=e.reader.old.score(p,rr);assert m['summary']==summary['summaries'][name]
    assert m['error_diagnostics']==summary['error_diagnostics'][name]
    independently=collections.Counter();typing=collections.Counter();per[name]=m['per_utterance']
    pp=[];nn=[];cl=collections.Counter();pc=[];query_probs=[];all_details=[]
    for r in rr:
        ref=norm(r['reference']);out=norm(r['prediction']);inp=norm(r['input']['asr_top1'])
        ops=alignment(ref,out); io=alignment(ref,inp);rp=pronunciation(ref);op=pronunciation(out)
        independently.update(chars=len(ref),errors=len(ops),input_errors=len(io),hits=sum(t in out for t in anns[r['id']]),mentions=len(anns[r['id']]))
        for kind,a,b in ops:
            typing[kind]+=1
            if kind=='S':
                key='same_phone_substitution' if rp[a]==op[b] else 'different_phone_substitution'
                typing[key]+=1
                if any(a in range(start,start+len(t)) for t in anns[r['id']] for start in range(len(ref)) if ref.startswith(t,start)):
                    typing['hotword_reference_substitution']+=1
        if name=='old_text':continue
        rec=slots[r['id']];lab=label_slots(dict(r,split='train'),rec);scores=r['compatibility_logits'];assert len(lab)==len(scores)
        offset=0;good=total=0
        for slot in rec['slots']:
            n=len(slot['groups']);ls=lab[offset:offset+n];ss=scores[offset:offset+n];offset+=n
            positive=[v for l,v in zip(ls,ss) if l==1];negative=[v for l,v in zip(ls,ss) if l==0]
            if positive and negative:total+=1;good+=max(positive)>max(negative)
        pc.append((good,total))
        for l,s in zip(lab,scores):
            if l<0:continue
            pr=1/(1+np.exp(-np.clip(s,-60,60)));query_probs.append(pr)
            if l==1:pp.append(pr);cl['TP' if s>=0 else 'FN']+=1
            else:nn.append(pr);cl['FP' if s>=0 else 'TN']+=1
        all_details.append(dict(id=r['id'],pair_correct=good,pair_total=total,logits=scores))
    assert (independently['chars'],independently['mentions'])==(21102,599)
    for field in ['chars','errors','input_errors','hits','mentions']:assert independently[field]==m['summary'][field],(name,field)
    audits[name]=dict(rows=len(rr),sha256=e.w.sha(file),independent_counts=dict(independently),error_types=dict(typing))
    metrics[name]=m['summary']
    if name!='old_text':
        tp,fn,tn,fp=(cl[x] for x in ['TP','FN','TN','FP']);gp=sum(x[0] for x in pc);gt=sum(x[1] for x in pc)
        classes[name]=dict(**cl,positive_recall=tp/(tp+fn),negative_recall=tn/(tn+fp),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
          mean_support_positive=statistics.mean(pp),mean_support_negative=statistics.mean(nn),support_quantiles=np.quantile(query_probs,[0,.1,.5,.9,1]).tolist(),pair_correct=gp,pair_total=gt,pair_accuracy=gp/gt)
        pair_vectors[name]=pc;details[name]=all_details

def comparison(a,b):
    result=collections.Counter();examples={'better':[],'worse':[],'equal_error_but_changed':[]}
    aa={x['id']:x for x in per[a]};bb={x['id']:x for x in per[b]}
    for x,y in zip(preds[a],preds[b]):
        uid=x['id'];delta=aa[uid]['errors']-bb[uid]['errors'];same=norm(x['prediction'])==norm(y['prediction'])
        result['changed']+=not same
        result['worse']+=delta>0;result['better']+=delta<0;result['equal_errors']+=delta==0
        result['added_errors']+=max(delta,0);result['removed_errors']+=max(-delta,0)
        result['baseline_exact_to_wrong']+=bb[uid]['errors']==0 and aa[uid]['errors']>0
        result['baseline_wrong_to_exact']+=bb[uid]['errors']>0 and aa[uid]['errors']==0
        if norm(x['reference'])==norm(x['input']['asr_top1']):
            result['delta_errors_on_correct_inputs']+=delta
        else:result['delta_errors_on_incorrect_inputs']+=delta
        cat='worse' if delta>0 else 'better' if delta<0 else 'equal_error_but_changed'
        if not same:
            examples[cat].append(dict(id=uid,reference=x['reference'],input=x['input']['asr_top1'],baseline=y['prediction'],output=x['prediction'],baseline_errors=bb[uid]['errors'],output_errors=aa[uid]['errors'],hotwords=anns[uid]))
    return dict(counts=dict(result),ci=p.bootstrap(per[a],per[b]),examples=examples)

comparisons={}
for a,b in [('CE_true','old_text'),('MATCH_true','old_text'),('MATCH_true','CE_true'),('CE_true','CE_neutral'),('MATCH_true','MATCH_neutral'),('MATCH_true','MATCH_zero'),('MATCH_true','MATCH_mismatch')]:
    comparisons[f'{a}_minus_{b}']=comparison(a,b)
for b,ci in summary['MATCH_true_minus_controls_95ci'].items():assert p.bootstrap(per['MATCH_true'],per[b])==ci

coverage=collections.Counter();old_wrong_coverage=collections.Counter();width=[];fallback=[];unknown=0
old={r['id']:r for r in preds['old_text']}
for r in rows:
    rec=slots[r['id']];ref=norm(r['reference']);rp=pronunciation(ref);base=rec['text'];bp=pronunciation(base)
    oldout=norm(old[r['id']]['prediction']);oldphones=pronunciation(oldout)
    gold_map=aligned_characters(base,ref);old_map=aligned_characters(base,oldout)
    if rec['timing_fallback']:fallback.append(r['id'])
    for slot,j,k in zip(rec['slots'],gold_map,old_map):
        width.append(slot['span'][1]-slot['span'][0]);unknown+=not slot['timing_anchored']
        if j is None:coverage['input_deleted_slots']+=1;continue
        aliases={a for g in slot['groups'] for a in g['aliases']};phones={g['phone'] for g in slot['groups']}
        coverage['aligned_gold_slots']+=1;coverage['gold_char_in_pool']+=ref[j] in aliases;coverage['gold_phone_in_pool']+=rp[j] in phones
        if base[slot['position']]!=ref[j]:
            kind='same_phone' if bp[slot['position']]==rp[j] else 'different_phone'
            coverage[f'input_{kind}_substitution']+=1;coverage[f'input_{kind}_gold_char_covered']+=ref[j] in aliases;coverage[f'input_{kind}_gold_phone_covered']+=rp[j] in phones
        if k is not None and oldout[k]!=ref[j]:
            kind='same_phone' if oldphones[k]==rp[j] else 'different_phone'
            old_wrong_coverage[f'{kind}_substitution']+=1;old_wrong_coverage[f'{kind}_gold_char_covered']+=ref[j] in aliases;old_wrong_coverage[f'{kind}_gold_phone_covered']+=rp[j] in phones
coverage['reference_insertions_missing_explicit_slots']=sum(sum(kind=='I' for kind,_,_ in alignment(norm(r['input']['asr_top1']),norm(r['reference']))) for r in rows)

def cluster_pair_ci(a,b):
    x=np.array(pair_vectors[a]);y=np.array(pair_vectors[b]);assert np.array_equal(x[:,1],y[:,1]);rng=np.random.default_rng(917);d=[]
    for _ in range(2000):
        ii=rng.integers(0,len(x),len(x));den=x[ii,1].sum()
        if den:d.append(100*(x[ii,0]-y[ii,0]).sum()/den)
    return np.quantile(d,[.025,.975]).tolist()
pair_ci={b:cluster_pair_ci('MATCH_true',b) for b in ['CE_true','MATCH_zero','MATCH_mismatch']}
result=dict(protocol_sha256=c['protocol_sha256'],initial_tensor_sha256=initial,audits=audits,metrics=metrics,
  classes=classes,pair_accuracy_delta_pp_95ci=pair_ci,comparisons=comparisons,
  input_coverage=dict(coverage),old_text_error_coverage_at_input_slots=dict(old_wrong_coverage),
  timing=dict(dev_uniform_fallback_ids=fallback,dev_unanchored_slots=unknown,window_frames_quantiles=np.quantile(width,[0,.1,.5,.9,1]).tolist(),warning='Input-only weak alignment; no gold boundaries, cannot establish temporal accuracy'),
  warnings=['Posthoc dev-reference/dictionary diagnostics only; no wave-verified phonetic labels','CER DP tie order diagonal deletion insertion','Phone coverage at input-aligned character positions misses insertion structure and uncertain alignment','Query accuracy has heavily imbalanced positives; balanced and mixed-slot rankings reported','Confidence intervals resample utterances; one training seed, no independence claim for repeated dev exploration'])
e.w.save(H/'posthoc_analysis_20261005.json',result)
print(json.dumps({k:result[k] for k in ['classes','pair_accuracy_delta_pp_95ci','input_coverage','old_text_error_coverage_at_input_slots','timing']},ensure_ascii=False),flush=True)
print('FULL_AUDIT_PASS',flush=True)
