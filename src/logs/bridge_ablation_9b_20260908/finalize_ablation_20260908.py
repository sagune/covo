"""Verify historical official metrics, then finalize fixed A/B/B+ comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/autodl-tmp')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'cbwhisper_covo_migration_20260609_tar_extracted/covo/src'))
from covo.metrics import edit_distance
from covo.text import normalize_chinese_text
from opencc import OpenCC
CC=OpenCC('t2s')
DATA=ROOT/'datasets/aishell/data_aishell_sensevoice/hotword/test'


def norm(s):
    return normalize_chinese_text(CC.convert(s))


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def metadata():
    utts=[x.split(maxsplit=1) for x in (DATA/'uttid').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    assert len(utts)==808 and len({x[0] for x in utts})==808
    aligned=[]
    for x in (DATA/'aligned.txt').read_text(encoding='utf-8-sig').splitlines():
        fields=x.split('\t')
        if len(fields)>=2:
            aligned.append((fields[1].strip(),norm(fields[0])))
    r1={norm(x.strip()) for x in (DATA/'r1-hotword.txt').read_text(encoding='utf-8-sig').splitlines() if x.strip()}
    assert len(aligned)==400
    assert sum(kw in r1 for uid,kw in aligned)==226
    return utts,aligned,r1


def metric(records,field='prediction'):
    utts,aligned,r1=metadata()
    byid={str(r['id']):r for r in records}
    assert len(byid)==808 and set(byid)=={str(i) for i in range(808)}
    pred={}
    edits=0
    chars=0
    for i,(uid,ref) in enumerate(utts):
        r=byid[str(i)]
        assert norm(r['reference'])==norm(ref),(i,uid)
        text=r['input']['asr_top1'] if field=='base' else r[field]
        pred[uid]=norm(text)
        edits+=edit_distance(list(norm(r['reference'])),list(pred[uid]))
        chars+=len(norm(r['reference']))
    hits=sum(kw in pred[uid] for uid,kw in aligned)
    r1hits=sum(kw in pred[uid] for uid,kw in aligned if kw in r1)
    return dict(cer=edits/chars,edits=edits,reference_chars=chars,recall400_hits=hits,
                recall400=hits/400,r1_hits=r1hits,r1_recall=r1hits/226,
                misses=[dict(id=uid,hotword=kw,prediction=pred[uid]) for uid,kw in aligned if kw not in pred[uid]])


def verify():
    source=ROOT/'src/logs/cb_sensevoice_acoustic_phrase_w14_covo_predictions_dpo30_aishell808_20260724.jsonl'
    cached=rows(source)
    base,pred=metric(cached,'base'),metric(cached)
    assert (base['edits'],base['recall400_hits'],base['r1_hits'])==(489,366,192),base
    assert (pred['edits'],pred['recall400_hits'],pred['r1_hits'])==(404,362,189),pred
    dump('official_metric_verification.json',dict(status='PASS',baseline=base,historical_dpo30=pred,
         metadata_sha256={str(DATA/name):hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['uttid','aligned.txt','r1-hotword.txt']},
         normalization='OpenCC t2s + project NFKC/whitespace/punctuation removal'))
    print('OFFICIAL_METRIC_VERIFIED',flush=True)


def finalize():
    data={v:rows(OUT/f'{v}.predictions.jsonl') for v in ['A','B','Bplus']}
    metrics={'CB':metric(data['A'],'base'),**{v:metric(rs) for v,rs in data.items()}}
    paired=[]
    for v in ['B','Bplus']:
        for a,b in zip(data['A'],data[v]):
            assert a['id']==b['id'] and a['input']==b['input'] and a['reference']==b['reference']
            ref=norm(a['reference'])
            ae=edit_distance(list(ref),list(norm(a['prediction'])))
            be=edit_distance(list(ref),list(norm(b['prediction'])))
            paired.append(dict(variant=v,id=a['id'],reference=a['reference'],A=a['prediction'],
                               output=b['prediction'],A_edits=ae,output_edits=be,delta=be-ae))
    dump('official_results.json',metrics)
    with (OUT/'paired_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    extra=json.loads((OUT/'results.json').read_text())
    lines=['# 9B 桥接 A/B/B+ 诊断实验','',
           '固定已有开发集选出的 checkpoint-30022，AISHELL-NE 808 条，batch=8，greedy，max_new_tokens=128，disable-thinking。',
           '已用这批测试结果设计提示，因此本轮不是新的无偏 held-out 证据；未训练、未选新 checkpoint。',
           '三组重新推理；A 是精确缓存原提示，B 只补词，B+ 补词并说明用法。候选及模型相同。','',
           '| 系统 | CER | Recall@400 | R1@226 |', '|---|---:|---:|---:|']
    for v,m in metrics.items():
        lines.append(f"| {v} | {100*m['cer']:.4f}% | {m['recall400_hits']}/400 ({100*m['recall400']:.2f}%) | {m['r1_hits']}/226 ({100*m['r1_recall']:.2f}%) |")
    lines+=['','## 相对 A 的逐句变化','', '| 系统 | 改善句 | 退化句 | CER 编辑数变化 | 新增非参考检索词项（相对 CB） |', '|---|---:|---:|---:|---:|']
    for v in ['B','Bplus']:
        cases=[r for r in paired if r['variant']==v]
        lines.append(f"| {v} | {sum(r['delta']<0 for r in cases)} | {sum(r['delta']>0 for r in cases)} | {sum(r['delta'] for r in cases):+d} | {extra[v].get('retrieved_nonreference_newly_inserted',0)} |")
    lines+=['','新增非参考检索词项是精确子串代理指标，包含标注不完整的可能，不等同于听音确认的幻觉。',
            '942 项 keyword_mentions 诊断另见 results.json，不与 Recall@400 混用。',
            '历史 DPO30 只用于评测器复现校验，不作为本轮 9B 的 A 组输出。',
            '清单和代码哈希、命令、时间、逐条预测、解析警告均在本目录留存。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump('FINALIZED.json',{'status':'complete','time':time.time()})
    print('FINALIZED',json.dumps({v:{k:x for k,x in m.items() if k!='misses'} for v,m in metrics.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    verify()
    if args.wait:
        deadline=time.time()+7200
        while not (OUT/'ALL_DONE.json').exists():
            if (OUT/'FAILED.json').exists():
                raise RuntimeError('Inference failed; inspect FAILED.json')
            if time.time()>deadline:
                raise TimeoutError('No inference completion within two hours')
            time.sleep(15)
        finalize()
