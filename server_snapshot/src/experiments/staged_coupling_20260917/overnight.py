#!/usr/bin/env python3
"""Finite overnight continuation; preserve the locked pilot and its decisions."""
import datetime as dt
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
PILOT=HERE/'run'
OUT=HERE/'overnight'
DEADLINE=dt.datetime(2026,9,17,10,0,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
spec=importlib.util.spec_from_file_location('pilot',HERE/'run.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def state(status,**kwargs):
    p.save(OUT/'status.json',dict(status=status,pid=os.getpid(),updated=time.time(),deadline=DEADLINE,**kwargs))


def prepare():
    assert not (OUT/'protocol.json').exists(),'independent overnight run already prepared'
    ev=p.evaluator()
    wavroot=p.ROOT/'datasets/aishell/data_aishell/wav'
    waves=sorted((wavroot/'train').rglob('*.wav'))
    ids={w.stem:w for w in waves}
    assert len(ids)==len(waves),'duplicate train utterance ID'
    # Only collect train references; do not read test references into this experiment.
    refs={}
    transcript=p.ROOT/'datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
    for line in transcript.open(encoding='utf-8-sig'):
        fields=line.strip().split(maxsplit=1)
        if len(fields)==2 and fields[0] in ids:
            assert fields[0] not in refs
            refs[fields[0]]=ev.norm(fields[1])
    dev=p.load(PILOT/'dev.jsonl')
    devrefs={ev.norm(r['reference']) for r in dev}
    official_dev_ids={w.stem for w in (wavroot/'dev').rglob('*.wav')}
    official_test_ids={w.stem for w in (wavroot/'test').rglob('*.wav')}
    assert set(ids).isdisjoint(official_dev_ids|official_test_ids)
    missing=[]; overlap=[]; rows=[]
    for uid,wav in ids.items():
        if not refs.get(uid):
            missing.append(uid)
        elif refs[uid] in devrefs:
            overlap.append(uid)
        else:
            rows.append(dict(id=uid,wav=str(wav),reference=refs[uid],split='train',
                             input={'asr_top1':''}))
    assert len(rows)>100000, 'unexpectedly incomplete full training source'
    rows.sort(key=lambda r:p.rank(r['id']))
    OUT.mkdir(exist_ok=True)
    p.save_rows(OUT/'full_train.jsonl',rows)
    for name in ['dev','screen']:
        shutil.copy2(PILOT/f'{name}.jsonl',OUT/f'{name}.jsonl')
    (OUT/'audio').symlink_to(PILOT/'audio',target_is_directory=True)
    p.save(OUT/'exclusions.json',dict(missing_or_empty_reference=missing,duplicate_reference_with_protocol_dev=overlap))
    pilot=json.loads((PILOT/'implementation_lock.json').read_text())
    assert p.digest(HERE/'run.py')==pilot['code_sha256']
    protocol=dict(created=time.time(),deadline=DEADLINE,deadline_local='2026-09-17 10:00 Asia/Shanghai',
        purpose='Independent expanded-data audio-only alignment continuation, NOT a reclassification of pilot failure',
        trigger='After original pipeline has clean terminal status and releases GPU; never override its gates',
        full_train_wave_n=len(waves),labelled_n=len(refs),eligible_train_n=len(rows),
        missing_reference_n=len(missing),duplicate_dev_reference_n=len(overlap),
        source_distribution='All eligible labelled train audio, not error-selected correction evidence',
        initialization=str(PILOT/'A_512'),fallback_initialization='none; missing checkpoint is an error',
        frozen='SenseVoice and original Qwen3.5-9B plus existing high_lr3e-5 checkpoint-625 LoRA',
        updated='connector only; fresh optimizer; warm start from pilot A_512, regardless of pilot positive/negative conclusion',
        max_epochs=1,microbatch=1,accumulation=8,lr=1e-4,warmup_steps=64,
        checkpoint_interval_steps=256,training_order='deterministic SHA256(917|uttid)',
        budget='Stop at next optimizer boundary at/after 10:00; save optimizer, connector, scheduler, RNG and next row. No new training after deadline.',
        evaluation='Original pilot retains its evaluations. Expanded run requires separate true/zero/mismatch screen evaluation before claiming any benefit; no automatic test tuning.',
        min_free_disk_GB=12,code_sha256=p.digest(__file__),pilot_code_sha256=p.digest(HERE/'run.py'),
        data_sha256=p.digest(OUT/'full_train.jsonl'),transcript_sha256=p.digest(transcript),
        cache='Reuse identical continuous frozen encoder caches; complete per-utterance .npy files are atomic',
        safety='No infinite retries, no dummy GPU work, no server shutdown; hardware/data faults stop for diagnosis')
    p.save(OUT/'protocol.json',protocol)
    state('PREPARED',eligible_train_n=len(rows),waiting_for_pilot=True)
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


def audit():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['code_sha256']==p.digest(__file__)
    assert protocol['pilot_code_sha256']==p.digest(HERE/'run.py')
    assert protocol['data_sha256']==p.digest(OUT/'full_train.jsonl')
    assert (OUT/'audio').resolve()==(PILOT/'audio').resolve()
    assert protocol['deadline']==DEADLINE
    return protocol


def worker_export():
    import numpy as np
    import torch
    from funasr import AutoModel
    from funasr.utils.load_utils import extract_fbank,load_audio_text_image_video
    audit()
    torch.set_num_threads(6)
    wrapper=AutoModel(model=str(p.SENSE),device='cuda:0',disable_update=True)
    model,frontend=wrapper.model.eval(),wrapper.kwargs['frontend']
    rows=p.load(OUT/'full_train.jsonl'); started=time.time(); new=0
    manifest=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if time.time()>=DEADLINE:
                state('BUDGET_REACHED_DURING_EXPORT',processed=i,total=len(rows),newly_exported=new)
                return
            dest=OUT/'audio'/f"{row['id']}.npy"
            if dest.exists():
                data=np.load(dest,mmap_mode='r')
                assert data.ndim==2 and data.shape[1]==512 and len(data)>0 and np.isfinite(data).all()
            else:
                if i%100==0:
                    assert shutil.disk_usage(OUT).free>12*10**9,'less than 12GB free disk; stop, do not delete old files'
                audio=load_audio_text_image_video(row['wav'],fs=frontend.fs,audio_fs=None,data_type='sound',tokenizer=None)
                speech,lengths=extract_fbank(audio,data_type='sound',frontend=frontend)
                speech,lengths=speech.cuda(),lengths.cuda()
                valid=int(lengths[0])
                def query(tokens):
                    return model.embed(torch.tensor([tokens],device='cuda',dtype=torch.long))
                x=torch.cat([query([model.lid_dict['zh']]),query([1,2]),query([model.textnorm_dict['woitn']]),speech],1)
                enc,out_lengths=model.encoder(x,lengths+4)
                if isinstance(enc,tuple): enc=enc[0]
                assert int(out_lengths[0])==valid+4 and enc.shape[1]>=valid+4
                data=enc[0,4:valid+4].float().cpu().numpy().astype(np.float16)
                assert data.shape==(valid,512) and np.isfinite(data).all()
                with open(str(dest)+'.overnight.tmp','wb') as f:
                    np.save(f,data)
                os.replace(str(dest)+'.overnight.tmp',dest)
                new+=1
            manifest.append(dict(id=row['id'],frames=len(data),sha256=p.digest(dest)))
            if (i+1)%500==0:
                state('EXPORTING',processed=i+1,total=len(rows),newly_exported=new,elapsed=time.time()-started)
                print(json.dumps(dict(done=i+1,total=len(rows),new=new,seconds=time.time()-started)),flush=True)
    p.save_rows(OUT/'audio_manifest.jsonl',manifest)
    p.save(OUT/'export_complete.json',dict(rows=len(rows),new=new,seconds=time.time()-started,
           manifest_sha256=p.digest(OUT/'audio_manifest.jsonl')))


def train():
    import torch
    import numpy as np
    protocol=audit()
    rows=p.load(OUT/'full_train.jsonl')
    exported=json.loads((OUT/'export_complete.json').read_text())
    assert exported['rows']==len(rows)
    assert (PILOT/'A_512'/'metadata.json').is_file()
    p.RUN=OUT
    model,tokenizer,connector=p.load_model('A',PILOT/'A_512',True)
    assert not any(v.requires_grad for v in model.parameters())
    optimizer=torch.optim.AdamW(connector.parameters(),lr=1e-4,weight_decay=.01)
    planned=math.ceil(len(rows)/8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/64,1.0))
    optimizer.zero_grad(set_to_none=True)
    started=time.time(); latest=None; seen=0; stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    def save_checkpoint(step,seen):
        dest=OUT/f'Afull_{step}'
        if not dest.exists():
            tmp=OUT/f'Afull_{step}.partial'
            tmp.mkdir(exist_ok=False)
            torch.save(connector.state_dict(),tmp/'connector.pt')
            torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                            numpy_rng=np.random.get_state(),python_rng=random.getstate(),next_row=seen,step=step),
                       tmp/'resume_state.pt')
            p.save(tmp/'metadata.json',dict(step=step,seen=seen,eligible=len(rows),complete_checkpoint=True,
                  source_checkpoint=str(PILOT/'A_512'),code_sha256=protocol['code_sha256'],data_sha256=protocol['data_sha256']))
            tmp.rename(dest)
        p.save(OUT/'latest_checkpoint.json',dict(path=str(dest),step=step,seen=seen,total=len(rows),updated=time.time()))
        return str(dest)
    for start in range(0,len(rows),8):
        if time.time()>=DEADLINE or stop_requested:
            break
        group=rows[start:start+8]; losses=[]
        for row in group:
            b=p.batch(model,tokenizer,connector,row,'A',training=True)
            loss=model(**b,use_cache=False).loss
            assert torch.isfinite(loss),f'nonfinite loss {row["id"]}'
            (loss/len(group)).backward()
            losses.append(float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(connector.parameters(),1.0)
        assert torch.isfinite(grad) and float(grad)>0
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        seen=start+len(group); step=(start//8)+1
        if step==1 or step%10==0:
            info=dict(step=step,planned_steps=planned,seen=seen,total=len(rows),loss=sum(losses)/len(losses),
                      grad_norm=float(grad),seconds=time.time()-started,gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            print(json.dumps(info),flush=True)
            state('TRAINING',**info)
        if step%256==0:
            latest=save_checkpoint(step,seen)
    if seen:
        latest=save_checkpoint(math.ceil(seen/8),seen)
    state('EPOCH_COMPLETED' if seen==len(rows) else 'BUDGET_CHECKPOINT_SAVED',seen=seen,total=len(rows),
          checkpoint=latest,seconds=time.time()-started,evaluation_pending=True)


def supervisor():
    audit()
    lock=open(OUT/'supervisor.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def child(name):
        assert time.time()<DEADLINE
        state('STARTING_'+name.upper())
        with open(OUT/f'{name}.log','x',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',__file__,name],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        while time.time()<DEADLINE:
            pilot=json.loads((PILOT/'status.json').read_text())
            status=pilot['status']
            if status=='FAILED':
                raise RuntimeError('Pilot failed technically; diagnose before any continuation: '+str(pilot))
            if status.startswith('COMPLETED'):
                break
            state('WAITING_FOR_PILOT',pilot_status=status,pilot_phase=pilot.get('phase'))
            time.sleep(30)
        if time.time()>=DEADLINE:
            state('DEADLINE_BEFORE_CONTINUATION',note='Original pilot not interrupted')
            return
        # The pilot may publish its terminal state moments before process exit.
        with open(PILOT/'pipeline.lock','r+') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            for _ in range(24):
                report=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if not report: break
                time.sleep(5)
            else:
                raise RuntimeError('Other GPU process still active; refusing concurrent allocation')
            assert (PILOT/'A_512'/'connector.pt').is_file()
            p.save(OUT/'parent_outcome.json',json.loads((PILOT/'status.json').read_text()))
            child('export')
            if time.time()<DEADLINE and (OUT/'export_complete.json').exists():
                child('train')
    except BaseException:
        state('FAILED',error=traceback.format_exc())
        raise


def selftest():
    protocol=audit()
    rows=p.load(OUT/'full_train.jsonl')
    p.unique(rows,'id')
    assert len(rows)==protocol['eligible_train_n']
    assert all(r['split']=='train' and r['reference'] and not r['input']['asr_top1'] for r in rows)
    assert set(r['id'] for r in rows).isdisjoint(r['id'] for r in p.load(PILOT/'dev.jsonl'))
    assert 'asr_top1' not in json.dumps(p.messages(rows[0],'A',True))
    assert p.MARKER in json.dumps(p.messages(rows[0],'A',True))
    assert dt.datetime.fromtimestamp(DEADLINE,dt.timezone(dt.timedelta(hours=8))).hour==10
    assert (OUT/'audio').resolve()==(PILOT/'audio').resolve()
    p.save(OUT/'selftest.json',dict(status='PASS',eligible_train_n=len(rows),code_sha256=p.digest(__file__),
        note='CPU audit; GPU primitives reuse unchanged, previously smoke-tested pilot implementation'))
    print('OVERNIGHT CPU SELFTEST PASS',flush=True)


if __name__=='__main__':
    {'prepare':prepare,'selftest':selftest,'export':worker_export,'train':train,'supervise':supervisor}[sys.argv[1]]()
