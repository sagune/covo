"""Post-training FULL/LOCAL evidence sensitivity on all TRAIN pairs, not dev selection."""
import sys
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
PARENT=Path('/root/autodl-tmp/src/experiments/evidence_decision_20261005_v3')
sys.path.insert(0,str(PARENT))
import experiment as ex
from interface import swap,order_bit,label_for,choose

def run():
    import torch
    c,d=ex.data_config();qs=ex.rows(PARENT/'train_queries.jsonl')
    assert len(qs)==5325 and len({q['edit_id'] for q in qs})==5325
    spec=ex.load(HERE/'queue_spec.json')
    assert ex.sha(PARENT/'protocol.json')==spec['parent_protocol_sha256']
    assert not (HERE/'train_diagnostic_seal.json').exists()
    hashes={};begin=time.time()
    # No reference or labels are read before all eight decision files are sealed.
    for arm in ('AUDIO','TEXT'):
        cp=PARENT/arm/'final_checkpoint';ex.verify_checkpoint(cp,c,arm)
        with ex.w.gpu_guard(ex.p),torch.inference_mode():
            rt=ex.Runtime(c,d,False,cp)
            for task in ('full','local'):
                for mode in ('true','none'):
                    key=arm+'__'+task+'__'+mode;file=HERE/(key+'.train_decisions.jsonl');partial=Path(str(file)+'.partial')
                    assert not file.exists() and not partial.exists()
                    with partial.open('x',encoding='utf-8') as f:
                        for i,q in enumerate(qs):
                            qq=swap(q) if order_bit(q['edit_id']) else q
                            ll=rt.logits(qq,mode,task).cpu().tolist();winner=choose(*ll)
                            out=dict(edit_id=q['edit_id'],id=q['id'],arm=arm,task=task,mode=mode,logits=ll,selected_key=winner,selected_role=qq[winner]['role'],batch_size=1,padding_tokens=0)
                            f.write(ex.json.dumps(out,ensure_ascii=False)+'\n');f.flush()
                            if (i+1)%250==0 or i+1==len(qs):
                                print(ex.json.dumps(dict(stage='TRAIN_SIDE_DIAGNOSIS',cell=key,done=i+1,total=5325,elapsed=time.time()-begin)),flush=True)
                    assert [x['edit_id'] for x in ex.rows(partial)]==[q['edit_id'] for q in qs]
                    partial.rename(file);hashes[key]=ex.sha(file)
            del rt
            torch.cuda.empty_cache()
    ex.save(HERE/'train_diagnostic_seal.json',dict(parent_protocol_sha256=ex.sha(PARENT/'protocol.json'),source_sha256=ex.sha(__file__),pairs=5325,cells=8,hashes=hashes,reference_used=False))
    # Training preference diagnosis only AFTER all raw decisions are immutable.
    labels={l['edit_id']:l for l in ex.rows(PARENT/'train_labels.jsonl')};tables={};pred={}
    for key,digest in hashes.items():
        file=HERE/(key+'.train_decisions.jsonl');assert ex.sha(file)==digest
        rr=ex.rows(file);ncorrect=0;nneutral=0;loss=0.
        for q,x in zip(qs,rr):
            assert x['edit_id']==q['edit_id'] and choose(*x['logits'])==x['selected_key']
            label=labels[q['edit_id']];y=label_for(q,label,bool(order_bit(q['edit_id'])))
            if label['neutral']:nneutral+=1
            else:ncorrect+=int((x['selected_key']=='A')==(y==1))
            a,b=x['logits'];peak=max(a,b);z=peak+ex.math.log(ex.math.exp(a-peak)+ex.math.exp(b-peak));loss+=z-y*a-(1-y)*b
        tables[key]=dict(pairs=5325,informative=5325-nneutral,neutral=nneutral,informative_correct=ncorrect,informative_accuracy=ncorrect/(5325-nneutral),mean_soft_ce=loss/5325)
        pred[key]=rr
    contrasts={}
    for arm in ('AUDIO','TEXT'):
        for task in ('full','local'):
            a=arm+'__'+task+'__true';b=arm+'__'+task+'__none'
            contrasts[a+'_vs_none']=dict(selection_changes=sum(x['selected_role']!=y['selected_role'] for x,y in zip(pred[a],pred[b])),pairs=5325)
        for mode in ('true','none'):
            a=arm+'__full__'+mode;b=arm+'__local__'+mode
            contrasts[a+'_vs_local']=dict(selection_changes=sum(x['selected_role']!=y['selected_role'] for x,y in zip(pred[a],pred[b])),pairs=5325)
    ex.save(HERE/'train_diagnostic_summary.json',dict(status='COMPLETE',pairs=5325,cells=8,summaries=tables,contrasts=contrasts,seal_sha256=ex.sha(HERE/'train_diagnostic_seal.json'),limitations=['Training-side seen candidates and labels, NOT generalization/CER evidence','No development-result-based tuning or checkpoint selection','Evidence-sensitivity correlations do not prove causal waveform understanding']))

if __name__=='__main__':run()
