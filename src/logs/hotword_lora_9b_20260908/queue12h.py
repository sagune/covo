"""Serial, deadline-bounded AISHELL research queue. Preserve all run artifacts."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import OUT,MODEL,ADAPTER,MIG,dump,formatted
from workflow import evaluate,rank

DEADLINE=datetime.fromisoformat('2026-09-09T04:05:00+00:00').timestamp()
RUNS=OUT/'scheduled'
RESULTS={}


def status(stage,**extra):
    dump('QUEUE_STATUS.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),
         deadline_utc=datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),**extra))
    print(stage,json.dumps(extra,ensure_ascii=True),flush=True)


def command(name,args,folder):
    folder.mkdir(parents=True,exist_ok=True)
    log=folder/(name+'.log')
    if log.exists():raise FileExistsError(log)
    (folder/(name+'.command.json')).write_text(json.dumps(args,indent=2),encoding='utf-8')
    status(name,folder=str(folder))
    with log.open('x',encoding='utf-8') as f:
        subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=MIG/'covo',
             env={**os.environ,'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})


def infer(name,data,adapter,folder,expected):
    dest=folder/(name+'.predictions.jsonl')
    if not dest.exists():
        command(name,[sys.executable,str(MIG/'covo/scripts/infer_lora_text.py'),
            '--input',str(data),'--output',str(dest),'--model-name-or-path',str(MODEL),
            '--adapter-path',str(adapter),'--batch-size','8','--max-new-tokens','128',
            '--temperature','0','--disable-thinking','--progress-every','128'],folder)
    m=evaluate(dest);assert m['samples']==expected,(name,m['samples'])
    (folder/(name+'.metrics.json')).write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    return m


def report():
    dump('QUEUE_RESULTS.json',RESULTS)
    lines=['# 12小时 AISHELL 热词 LoRA 实验','',
      '固定5000条训练数据；开发集1334条，所有训练从同一个原LoRA开始。结果用于开发与消融，不是新测试集结果。',
      '统一比较CER、指定热词召回、纠错丢词/补词及非参考检索词误插入代理。','',
      '| 实验 | CER | 指定热词召回 | 丢词 | 补词 |','|---|---:|---:|---:|---:|']
    for name,r in RESULTS.items():
        m=r['metrics'];lines.append(f"| {name} | {100*m['cer']:.4f}% | {m['designated_hits']}/{m['designated_total']} ({100*m['recall']:.2f}%) | {m['hotwords_lost']} | {m['hotwords_gained']} |")
    lines+=['','截止时间为北京时间2026-09-09 12:05；不自动关机。',
      '上下文删除实验仅删除纠错输入中的KWS/prompt热词，CB候选保持缓存值，因此不能代表重新运行无热词的前端识别。']
    (OUT/'QUEUE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def select(runname,folder,steps):
    checks=sorted(set([math.ceil(steps/2),steps]))
    for step in checks:
        infer(f'screen_{step}',OUT/'dev.screen.unified.jsonl',folder/f'adapter/checkpoint-{step}',folder,334)
    # Two checkpoints: evaluate both fully, following the fixed screen/full-dev protocol.
    full={}
    for step in checks:
        full[f'checkpoint-{step}']=infer(f'full_{step}',OUT/'dev.unified.jsonl',folder/f'adapter/checkpoint-{step}',folder,1334)
    ranking=sorted(full.items(),key=rank);key,m=ranking[0]
    choice=dict(adapter=str(folder/'adapter'/key),metrics=m,selection='dev only',recall_floor_met=m['recall']>=.9)
    (folder/'selected_checkpoint.json').write_text(json.dumps(choice,indent=2),encoding='utf-8')
    RESULTS[runname]=choice;report()


def main():
    RUNS.mkdir(exist_ok=True)
    dump('QUEUE_PLAN.json',dict(deadline_utc='2026-09-09T04:05:00Z',base=str(MODEL),starting_adapter=str(ADAPTER),
      phases=['finish active 5000-row training','format baseline comparisons','lower learning rate','independent seed','two epochs if time permits','higher learning rate / third seed if time permits','context-omission robustness'],
      data_frozen=True,training_seed_default=20260908,test_for_selection=False,
      scheduling='serial GPU use; conservative time estimate before each new training; active stage may finish after deadline; no server shutdown'))
    status('waiting_for_active_training')
    while not (OUT/'TRAIN_DONE.json').exists():
        if not Path('/proc/9221').exists():raise RuntimeError('Active training exited without TRAIN_DONE.json; inspect train.log')
        if time.time()>DEADLINE:raise TimeoutError('Active training exceeded queue window')
        time.sleep(20)
    time.sleep(10)
    for name,filename in [('old_adapter_old_format','dev.old.jsonl'),('old_adapter_unified','dev.unified.jsonl')]:
        m=infer(name,OUT/filename,ADAPTER,OUT,1334)
        RESULTS[name]=dict(adapter=str(ADAPTER),metrics=m);report()
    select('initial_lr2e-5_seed20260908',OUT,625)
    tasks=[('low_lr1e-5',1e-5,1,20260908),('seed20260909',2e-5,1,20260909),
           ('two_epochs',2e-5,2,20260908),('high_lr3e-5',3e-5,1,20260908),
           ('seed20260910',2e-5,1,20260910)]
    skipped=[]
    for name,lr,epochs,seed in tasks:
        remaining=DEADLINE-time.time()
        expected=625*epochs*10.5+1500
        if remaining<expected:
            skipped.append(dict(name=name,remaining=remaining,estimated=expected));continue
        folder=RUNS/name
        command('train',[sys.executable,str(OUT/'train.py'),'--run-dir',str(folder),
            '--learning-rate',str(lr),'--epochs',str(epochs),'--seed',str(seed)],folder)
        assert (folder/'TRAIN_DONE.json').exists()
        select(name,folder,625*epochs)
    dump('QUEUE_SKIPPED_BUDGET.json',skipped)
    trained={k:v for k,v in RESULTS.items() if not k.startswith('old_adapter')}
    best=sorted(trained.items(),key=lambda kv:rank((kv[0],kv[1]['metrics'])))[0]
    dump('QUEUE_SELECTED_DEV.json',dict(name=best[0],**best[1]))
    # Useful shorter jobs occupy remaining time and characterize reliance on context.
    source=[json.loads(x) for x in (OUT/'dev.unified.jsonl').read_text().splitlines()]
    for fraction in [0.5,0.0,0.25]:
        for label,adapter in [('old',ADAPTER),('selected',Path(best[1]['adapter']))]:
            if DEADLINE-time.time()<550:break
            folder=RUNS/f'context_keep_{fraction}_{label}';folder.mkdir(parents=True,exist_ok=True)
            data=folder/'messages.jsonl'
            if data.exists():raise FileExistsError(data)
            with data.open('x',encoding='utf-8') as f:
                for original in source:
                    r=copy.deepcopy(original)
                    for field in ['hotwords','prompt_hotwords']:
                        kept=[]
                        for h in r['input'].get(field,[]):
                            score=int(hashlib.sha256((r['id']+':'+h['text']).encode()).hexdigest()[:8],16)/2**32
                            if score<fraction:kept.append(h)
                        r['input'][field]=kept
                    r=formatted(r,True)
                    f.write(json.dumps(r,ensure_ascii=False)+'\n')
            m=infer('infer',data,adapter,folder,1334)
            RESULTS[folder.name]=dict(adapter=str(adapter),metrics=m);report()
    report();status('queue_complete',results=len(RESULTS),seconds_remaining=max(0,DEADLINE-time.time()))
    dump('QUEUE_DONE.json',dict(time=time.time(),results=len(RESULTS)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        dump('QUEUE_FAILED.json',dict(error=repr(error),time=time.time()))
        status('failed',error=repr(error))
        raise
