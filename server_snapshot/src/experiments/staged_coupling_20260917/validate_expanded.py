#!/usr/bin/env python3
"""Locked-checkpoint validation; no training or checkpoint selection."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

HERE=Path(__file__).resolve().parent
PILOT=HERE/'run'
OUT=HERE/'validation_Afull_3243_full_dev_20260917'
CHECKPOINT=HERE/'overnight/Afull_3243'
spec=importlib.util.spec_from_file_location('pilot',HERE/'run.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def main():
    OUT.mkdir(exist_ok=False)
    lock=open(OUT/'validation.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    try:
        original=json.loads((PILOT/'implementation_lock.json').read_text())
        assert p.digest(HERE/'run.py')==original['code_sha256']
        protocol=json.loads((PILOT/'protocol.json').read_text())
        assert p.digest(PILOT/'dev.jsonl')==protocol['data_hashes']['dev']
        rows=p.load(PILOT/'dev.jsonl'); p.unique(rows,'id'); assert len(rows)==1334
        assert set(r['id'] for r in rows).isdisjoint(r['id'] for r in p.load(HERE/'overnight/full_train.jsonl'))
        mapping=p.permutation(rows)
        for path in [p.ADAPTER/'adapter_model.safetensors',p.ADAPTER/'adapter_config.json']:
            assert p.digest(path)==protocol['sources'][str(path)]
        original_records=p.load(p.DEV_SOURCE)
        assert [r['id'] for r in original_records]==[r['id'] for r in rows]
        assert [r['reference'] for r in original_records]==[r['reference'] for r in rows]
        config=dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=p.digest(CHECKPOINT/'connector.pt'),
            pilot_implementation_sha256=p.digest(HERE/'run.py'),validation_code_sha256=p.digest(__file__),
            dev_sha256=p.digest(PILOT/'dev.jsonl'),n=1334,selection='fixed latest overnight checkpoint; no sweep',
            tasks={'text':'fresh matched-prompt original frozen checkpoint-625 text-only baseline',
                   'A_before':'fresh full-dev audio-only pilot A_512 reference result',
                   'A':'audio-only transcription, direct before/after comparison with pilot A_512',
                   'B_probe':'untrained transfer probe: audio plus text, original frozen checkpoint-625 LoRA; NOT stage-B-trained model'},
            modes=['true','zero','mismatch'],decode=protocol['decode'],max_new_tokens=96,
            normalization=protocol['normalization'],audio_permutation=mapping,
            result_label='All 1334 protocol dev utterances; diagnostic full protocol dev, no independent test result',
            stopping='Any ID/cache/generation/runtime fault aborts; malformed JSON scored empty, no dropped utterances',
            metrics='corpus CER, designated hits/mentions, lost/gained, improved/worsened/unchanged; paired bootstrap')
        p.save(OUT/'protocol.json',config)
        p.save(OUT/'status.json',dict(status='RUNNING',pid=os.getpid(),started=started))
        model,tokenizer,connector=p.load_model('A',CHECKPOINT,False)
        # Keep p.RUN at PILOT so acoustic caches and shared metric functions stay identical.
        import torch
        results={}
        for task in ['text','A_before','A','B_probe']:
            active_checkpoint=PILOT/'A_512' if task in ['text','A_before'] else CHECKPOINT
            connector.load_state_dict(torch.load(active_checkpoint/'connector.pt',map_location='cuda',weights_only=True))
            results[task]={}
            modes=['text'] if task=='text' else ['true'] if task=='A_before' else ['true','zero','mismatch']
            for mode in modes:
                dest=OUT/f'{task}_{mode}.predictions.jsonl'
                predictions=[]
                p.save(OUT/'status.json',dict(status='RUNNING',task=task,mode=mode,pid=os.getpid(),started=started))
                with open(str(dest)+'.partial','x',encoding='utf-8') as f:
                    for i,row in enumerate(rows):
                        pred=p.generate(model,tokenizer,connector,row,'A' if task in ['A','A_before'] else 'B',mode,
                             mapping[row['id']] if mode=='mismatch' else None)
                        predictions.append(pred)
                        f.write(json.dumps(pred,ensure_ascii=False)+'\n'); f.flush()
                        if (i+1)%50==0:
                            print(json.dumps(dict(task=task,mode=mode,done=i+1,total=len(rows),seconds=time.time()-started)),flush=True)
                checked=p.load(str(dest)+'.partial')
                assert [r['id'] for r in checked]==[r['id'] for r in rows]
                assert len(checked)==1334
                os.replace(str(dest)+'.partial',dest)
                metric=p.metrics(checked)
                p.save(OUT/f'{task}_{mode}.metrics.json',metric)
                results[task][mode]=metric['summary']
                print(json.dumps(dict(task=task,mode=mode,metrics=metric['summary'])),flush=True)
                if task in ['B_probe','text']:
                    protected=p.metrics(checked,True)
                    p.save(OUT/f'{task}_{mode}.protected.metrics.json',protected)
                    results[task][mode+'_protected']=protected['summary']
        old=json.loads((OUT/'A_before_true.metrics.json').read_text())
        new=json.loads((OUT/'A_true.metrics.json').read_text())
        results['A_before_summary']=old['summary']
        results['A_change_bootstrap']=p.bootstrap(new['per_utterance'],old['per_utterance'])
        a=results['A']
        results['A_audio_use_gate']=a['true']['parse_failures']==0 and a['true']['empty']==0 and a['true']['cer']<min(a['zero']['cer'],a['mismatch']['cer'])
        results['A_improved_over_pilot']=a['true']['cer']<old['summary']['cer']
        b=results['B_probe']
        for name,file in [('matched_text',OUT/'text_text.metrics.json'),
                          ('matched_protected_text',OUT/'text_text.protected.metrics.json'),
                          ('legacy_protected_text',PILOT/'legacy_dev_text.protected.metrics.json')]:
            base=json.loads(file.read_text())
            results[name]=base['summary']
            assert [x['id'] for x in base['per_utterance']]==[r['id'] for r in rows]
        results['B_probe_meets_joint_objective']=b['true']['parse_failures']==0 and b['true']['empty']==0 and \
            b['true']['cer']<min(b['zero']['cer'],b['mismatch']['cer']) and \
            all(b['true']['cer']<results[k]['cer'] and b['true']['hits']>=results[k]['hits']
                for k in ['matched_protected_text','legacy_protected_text'])
        results['B_probe_vs_matched_protected_bootstrap']=p.bootstrap(
            json.loads((OUT/'B_probe_true.metrics.json').read_text())['per_utterance'],
            json.loads((OUT/'text_text.protected.metrics.json').read_text())['per_utterance'])
        results['seconds']=time.time()-started
        p.save(OUT/'summary.json',results)
        p.save(OUT/'status.json',dict(status='COMPLETED',seconds=time.time()-started,
               A_audio_use_gate=results['A_audio_use_gate'],A_improved_over_pilot=results['A_improved_over_pilot'],
               B_probe_meets_joint_objective=results['B_probe_meets_joint_objective']))
    except BaseException:
        p.save(OUT/'status.json',dict(status='FAILED',seconds=time.time()-started,error=traceback.format_exc()))
        raise


if __name__=='__main__': main()
