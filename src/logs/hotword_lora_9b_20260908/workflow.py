"""Persistent small-LoRA pilot: prepare, train, dev-only format/adapter comparisons."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import ROOT,OUT,MIG,MODEL,ADAPTER,DATA,dump,norm,sha,mapping

sys.path.insert(0,str(MIG/'covo/src'))
from covo.metrics import edit_distance


def run(name,command,cwd=None):
    print('START',name,time.strftime('%F %T'),flush=True)
    dump(name+'.command.json',command)
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:
        subprocess.run(command,cwd=cwd or ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
    print('DONE',name,time.strftime('%F %T'),flush=True)


def evaluate(path):
    records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    refs={uid:norm(ref) for uid,ref in mapping('dev')}
    designated=[]
    for line in (DATA/'dev/aligned.txt').read_text(encoding='utf-8-sig').splitlines():
        fs=line.split('\t')
        if len(fs)>=2:designated.append((fs[1].strip(),norm(fs[0])))
    byid={r['id']:r for r in records};assert len(byid)==len(records)
    edits=base_edits=chars=improved=worsened=exact=warnings=lost=gained=inserted=0
    for r in records:
        assert norm(r['reference'])==refs[r['id']]
        gold,base,pred=refs[r['id']],norm(r['input']['asr_top1']),norm(r['prediction'])
        bd=edit_distance(list(gold),list(base));pd=edit_distance(list(gold),list(pred))
        edits+=pd;base_edits+=bd;chars+=len(gold);improved+=pd<bd;worsened+=pd>bd;exact+=pd==0
        warnings+=bool(r.get('parse_warnings'))
        inserted+=sum(kw not in gold and kw not in base and kw in pred for kw in {norm(x['text']) for x in r['input']['hotwords']} if kw)
    active=[(uid,kw) for uid,kw in designated if uid in byid]
    hits=base_hits=0
    for uid,kw in active:
        r=byid[uid];bh=kw in norm(r['input']['asr_top1']);ph=kw in norm(r['prediction'])
        hits+=ph;base_hits+=bh;lost+=bh and not ph;gained+=ph and not bh
    assert chars and active
    return dict(samples=len(records),cer=edits/chars,baseline_cer=base_edits/chars,
        edits=edits,reference_chars=chars,improved=improved,worsened=worsened,unchanged=len(records)-improved-worsened,
        exact=exact,parse_warning_samples=warnings,designated_hits=hits,designated_total=len(active),
        recall=hits/len(active),baseline_recall=base_hits/len(active),hotwords_lost=lost,hotwords_gained=gained,
        nonreference_retrieved_new_insertions=inserted,normalization='OpenCC t2s + NFKC + punctuation/space removal')


def infer(name,inputfile,adapter,expected):
    path=OUT/(name+'.predictions.jsonl')
    run(name,[sys.executable,str(MIG/'covo/scripts/infer_lora_text.py'),
        '--input',str(OUT/inputfile),'--output',str(path),'--model-name-or-path',str(MODEL),
        '--adapter-path',str(adapter),'--batch-size','8','--max-new-tokens','128','--temperature','0',
        '--disable-thinking','--progress-every','128'],MIG/'covo')
    m=evaluate(path);assert m['samples']==expected
    dump(name+'.metrics.json',m)
    return m


def rank(item):
    name,m=item
    return (0,m['cer'],-m['recall'],name) if m['recall']>=.9 else (1,-m['recall'],m['cer'],name)


def main():
    started=time.time()
    deadline=started+7200
    while not (OUT/'dev_evidence.exit').exists():
        if time.time()>deadline:raise TimeoutError('Dev evidence did not finish within 2h')
        time.sleep(15)
    assert (OUT/'dev_evidence.exit').read_text().strip()=='exit=0'
    run('prepare',[sys.executable,str(OUT/'prepare.py')])
    run('train',[sys.executable,str(OUT/'train.py')])
    assert (OUT/'TRAIN_DONE.json').exists()
    # No test metrics or test inputs used in checkpoint selection.
    full={}
    full['old_adapter_old_format']=infer('old_adapter_old_format','dev.old.jsonl',ADAPTER,1334)
    full['old_adapter_unified']=infer('old_adapter_unified','dev.unified.jsonl',ADAPTER,1334)
    screen={}
    for step in [313,625]:
        screen[f'checkpoint-{step}']=infer(f'screen_{step}','dev.screen.unified.jsonl',OUT/f'adapter/checkpoint-{step}',334)
    dump('screen_ranking.json',sorted(screen.items(),key=rank))
    # There are only two saved checkpoints, so both receive full-dev evaluation.
    candidates={}
    for name,_ in sorted(screen.items(),key=rank):
        candidates[name]=infer('full_'+name,'dev.unified.jsonl',OUT/'adapter'/name,1334)
    ranking=sorted(candidates.items(),key=rank);selected=ranking[0][0]
    dump('full_dev_ranking.json',ranking)
    dump('selected_checkpoint.json',dict(path=str(OUT/'adapter'/selected),metrics=candidates[selected],
         meets_recall_floor=candidates[selected]['recall']>=.9,selection_split='dev only',
         selection_rule='recall>=0.90: lowest CER; otherwise highest recall then lowest CER',
         classification='small continuation pilot; dev-selected, not held-out test result'))
    full.update(candidates);dump('results.json',full)
    lines=['# AISHELL 9B 热词保留 LoRA 小规模实验','',
      '训练 5000 条、1 epoch；延续旧 LoRA，基座冻结，assistant-only loss。全部结果为开发集结果，不是新 held-out 结果。',
      '训练样本按官方 utterance ID 和归一化参考文本排除 dev/test 重叠；gold 仅用于训练目标、分层及评测，不传入 user 消息。','',
      '| 方案 | CER | 热词召回 | 丢词 | 补词 | 误插词代理 |','|---|---:|---:|---:|---:|---:|']
    for name,m in full.items():lines.append(f"| {name} | {100*m['cer']:.4f}% | {m['designated_hits']}/{m['designated_total']} ({100*m['recall']:.2f}%) | {m['hotwords_lost']} | {m['hotwords_gained']} | {m['nonreference_retrieved_new_insertions']} |")
    lines+=['',f'开发集选中：{selected}。',
      '误插词为检索词不在参考文本却新出现在输出的精确子串代理，不等同于听音确认的幻觉。',
      '旧模型两种输入只改桥接格式；LoRA 与统一格式对比用于隔离训练作用。',
      '此前已用于诊断的 AISHELL-NE 808 条没有用于训练或选 checkpoint，也没有自动重跑测试。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump('ALL_DONE.json',dict(elapsed_seconds=time.time()-started,selected=selected))


if __name__=='__main__':
    try:main()
    except Exception as error:
        dump('FAILED.json',dict(error=repr(error),time=time.time()))
        raise
