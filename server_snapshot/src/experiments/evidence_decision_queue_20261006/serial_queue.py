"""Finite, fail-closed serial queue; non-stdlib filename to avoid import shadowing."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def status(stage,**kw):
    obj=dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    tmp=HERE/'queue_status.tmp';tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(HERE/'queue_status.json')
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def check_sources(spec):
    for path,digest in spec['sources'].items():assert sha(path)==digest,path

def identity(pid):
    path=Path('/proc')/str(pid)
    if not path.exists():return None
    # starttime protects against PID recycling; state Z is terminated.
    stat=(path/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    if tail[0]=='Z':return None
    return dict(start_ticks=int(tail[19]),cmdline=(path/'cmdline').read_bytes().replace(b'\x00',b' ').decode())

def completed(directory):
    directory=Path(directory);c=load(directory/'protocol.json');d=load(directory/'data_lock.json');s=load(directory/'summary.json')
    ph=sha(directory/'protocol.json');dh=sha(directory/'data_lock.json')
    assert load(directory/'status.json')['stage']=='ALL_DONE'
    assert s['status']=='COMPLETE' and s['protocol_sha256']==ph and s['data_lock_sha256']==dh and d['protocol_sha256']==ph
    for path,digest in c['sources'].items():assert sha(path)==digest,path
    for path,digest in d['hashes'].items():assert sha(path)==digest,path
    ids=[x['id'] for x in rows(c['baseline'])];edit_ids=[q['edit_id'] for q in rows(directory/'dev_queries.jsonl')]
    assert len(ids)==len(set(ids))==1334 and len(edit_ids)==190
    assert len(s['summaries'])==12 and len(s['prediction_hashes'])==12
    for arm in ('UNADAPTED','AUDIO','TEXT'):
        seal=load(directory/arm/'decision_seal.json')
        assert seal['protocol_sha256']==ph and seal['data_lock_sha256']==dh and seal['reference_used'] is False
        assert set(seal['hashes'])=={'none','true','zero','mismatch','true_reversed'}
        for mode,digest in seal['hashes'].items():
            file=directory/arm/(mode+'.decisions.jsonl');assert sha(file)==digest
            assert [x['edit_id'] for x in rows(file)]==edit_ids
        for mode in ('none','true','zero','mismatch'):
            key=arm+'__'+mode;metric=s['summaries'][key]
            assert (metric['n'],metric['chars'],metric['mentions'])==(1334,21102,599)
            file=directory/arm/(mode+'.predictions.jsonl');assert sha(file)==s['prediction_hashes'][key]
            assert [x['id'] for x in rows(file)]==ids
        if arm!='UNADAPTED':
            cp=directory/arm/'final_checkpoint';meta=load(cp/'metadata.json')
            assert meta['complete'] and (meta['step'],meta['examples'],meta['arm'])==(666,5325,arm)
            assert meta['protocol_sha256']==ph and meta['data_lock_sha256']==dh
            assert meta['initial_adapter_hash']==sha(Path(c['checkpoint'])/'adapter/adapter_model.safetensors')
            for name,digest in meta['hashes'].items():assert sha(cp/name)==digest
    return dict(protocol_sha256=ph,data_lock_sha256=dh,summary_sha256=sha(directory/'summary.json'),groups=12,n=1334)

def run(spec):
    check_sources(spec)
    status('WAITING_CURRENT',parent=spec['parent'],parent_pid=spec['parent_pid'],planned_seeds=[918,919],deadline_epoch=spec['deadline_epoch'])
    while True:
        now=identity(spec['parent_pid'])
        if now is None:break
        assert now==spec['parent_identity'],'Parent PID reused or command identity changed'
        parent_status=load(Path(spec['parent'])/'status.json')
        assert parent_status['stage']!='FAILED','Current locked run failed; no automatic replacement'
        time.sleep(15)
    audited=[dict(path=spec['parent'],audit=completed(spec['parent']))]
    (HERE/'completion_audits.json').write_text(json.dumps(audited,indent=2),encoding='utf-8')
    for job in spec['jobs']:
        check_sources(spec)
        if time.time()>=spec['deadline_epoch']:
            status('WINDOW_REACHED',not_started=job,completed=audited);return
        assert shutil.disk_usage(HERE).free>8*1024**3,'Insufficient free disk; retain all old files'
        log=HERE/job['log']
        with log.open('x',encoding='utf-8') as f:
            proc=subprocess.Popen(job['command'],cwd=job['cwd'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            status('RUNNING_JOB',job=job['name'],child_pid=proc.pid,command=job['command'],completed=audited)
            rc=proc.wait()
        assert rc==0,f"Queue job failed: {job['name']} exit={rc} log={log}"
        if job['kind']=='repeat':audit=completed(job['cwd'])
        else:
            audit=load(HERE/'train_diagnostic_summary.json')
            assert audit['status']=='COMPLETE' and audit['pairs']==5325 and audit['cells']==8
        audited.append(dict(path=job['cwd'],name=job['name'],audit=audit))
        (HERE/'completion_audits.json').write_text(json.dumps(audited,indent=2),encoding='utf-8')
    status('QUEUE_COMPLETE',completed=audited,elapsed_seconds=time.time()-spec['start_epoch'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    spec=load(HERE/'queue_spec.json');check_sources(spec)
    if a.preflight:
        assert identity(spec['parent_pid'])==spec['parent_identity']
        assert len(spec['jobs'])==3
        print('QUEUE_PREFLIGHT_PASS_NO_GPU_LAUNCH',flush=True)
    else:
        # Exclusive instance guard; training children retain original gpu_guard.
        with (HERE/'queue.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not (HERE/'queue_status.json').exists(),'Queue already initialized; do not resubmit'
            try:run(spec)
            except BaseException:
                status('QUEUE_FAILED',error=traceback.format_exc());raise
