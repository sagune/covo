"""Serialized, fail-closed full-data pipeline; never reduces the dataset."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from prepare_full import legacy

HERE = Path(__file__).resolve().parent
PYTHON = '/root/autodl-tmp/great/bin/python'


def run(export_pid, skip_frontend=False):
    p = legacy()
    with (HERE / 'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (HERE / 'audio_export_complete.json').exists():
            cmdline = Path(f'/proc/{export_pid}/cmdline')
            assert cmdline.exists() and b'prepare_full.py' in cmdline.read_bytes(), 'Audio export stopped without completion'
            p.save(HERE / 'queue_status.json',dict(status='WAITING_FOR_AUDIO_EXPORT',pid=os.getpid(),export_pid=export_pid,time=time.time()))
            time.sleep(15)
        stages = [] if skip_frontend else [('frontend',[PYTHON,'-u',str(HERE/'build_evidence.py')])]
        for architecture in ('residual','mlp'):
            for command in ('prepare','smoke','train'):
                stages.append((architecture+'_'+command,[PYTHON,'-u',str(HERE/'full_workflow.py'),command,architecture]))
        for architecture in ('residual','mlp'):
            stages.append((architecture+'_eval',[PYTHON,'-u',str(HERE/'full_workflow.py'),'eval',architecture]))
        for stage,command in stages:
            marker = HERE / f'queue_{stage}_complete.json'
            if marker.exists():
                continue
            logpath = HERE / f'{stage}_{time.time_ns()}.log'
            with logpath.open('x') as log:
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                p.save(HERE / 'queue_status.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),child_pid=child.pid,command=command,log=str(logpath),time=time.time()))
                code = child.wait()
            if code:
                p.save(HERE / 'queue_status.json',dict(status='STOPPED_ON_FAILURE',stage=stage,exit_code=code,log=str(logpath),time=time.time()))
                raise RuntimeError(f'{stage} failed with code {code}; no automatic protocol changes')
            p.save(marker,dict(stage=stage,log=str(logpath),time=time.time()))
        for architecture in ('residual','mlp'):
            summary = HERE / 'runs' / architecture / 'official_full_dev/summary.json'
            assert summary.is_file()
        p.save(HERE / 'queue_status.json',dict(status='COMPLETED_BOTH_FULL_EPOCH_AND_OFFICIAL_DEV',time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-pid',required=True,type=int)
    parser.add_argument('--skip-frontend',action='store_true')
    args = parser.parse_args()
    try:
        run(args.export_pid, args.skip_frontend)
    except BaseException:
        legacy().save(HERE / f'queue_failure_{time.time_ns()}.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
