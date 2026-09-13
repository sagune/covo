"""Prepare train-only continued-LoRA data and independent contextual dev inputs."""
import argparse
import collections
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp')
OUT=Path(__file__).resolve().parent
MIG=ROOT/'cbwhisper_covo_migration_20260609_tar_extracted'
DATA=ROOT/'datasets/aishell/data_aishell_sensevoice/hotword'
MODEL=MIG/'models/Qwen3.5-9B'
ADAPTER=MIG/'covo/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022'


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


bridge=module('bridge',ROOT/'src/analysis/bridge_b_prompt_v2_20260907/cbsensevoice_covo_bridge.py')
norm=bridge.normalize_text


def mapping(split):
    ids=[line.strip().split()[0] for line in (DATA/split/'uttid').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    # AISHELL dev/test uttid files contain IDs only; transcripts live in text.
    refs={line.split(maxsplit=1)[0]:line.split(maxsplit=1)[1]
          for line in (DATA/split/'text').read_text(encoding='utf-8-sig').splitlines()
          if line.strip() and len(line.split(maxsplit=1))==2}
    if len(ids)!=len(refs) or set(ids)!=set(refs):
        raise ValueError(f'{split}: uttid/text mismatch ids={len(ids)} refs={len(refs)}')
    return [(uid,refs[uid]) for uid in ids]


def options(enabled):
    p=argparse.ArgumentParser();bridge.add_prepare_args(p)
    return p.parse_args(['--input','unused','--output','unused','--max-nbest','6','--max-pinyin','3',
                         '--include-pinyin','--hotword-source','all','--protect-supported-hotwords']+
                        (['--include-missing-kws-hotwords'] if enabled else []))


def formatted(r,enabled,training=False):
    audit={};args=options(enabled)
    # Gold fields are not passed to the prompt builder.
    safe={'input':{k:v for k,v in r['input'].items() if k not in ['keyword_mentions','oracle_hotwords']}}
    user=bridge.build_user_prompt(safe,args,audit)
    out={k:copy.deepcopy(r[k]) for k in ['id','split','reference','input']}
    out['messages']=[{'role':'system','content':bridge.build_system_message(args)},{'role':'user','content':user}]
    if training:
        out['messages'].append({'role':'assistant','content':json.dumps({'text':bridge.simplify_text(r['reference'])},ensure_ascii=False)})
    if enabled:out['hotword_transmission']=audit
    return out


def write(name,rows):
    with (OUT/name).open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def main():
    from transformers import AutoTokenizer
    if (OUT/'data_manifest.json').exists():raise FileExistsError('Do not overwrite prepared data')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    devmap,testmap=mapping('dev'),mapping('test')
    blocked_ids={x[0] for x in devmap+testmap};blocked_refs={norm(x[1]) for x in devmap+testmap}
    maps={};pool=[];seen=set();excluded=collections.Counter()
    source=ROOT/'src/logs/cb_sensevoice_w14_train_evidence_full.jsonl.gz'
    with gzip.open(source,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);split=r['split'];assert split.startswith('train_covo_shard')
            if split not in maps:maps[split]=mapping(split)
            uid,ref=maps[split][int(r['id'])];assert norm(ref)==norm(r['reference'])
            assert not r['input'].get('oracle_hotwords')
            if uid in blocked_ids or norm(ref) in blocked_refs:excluded['dev_test_overlap']+=1;continue
            if norm(ref) in seen:excluded['duplicate_reference']+=1;continue
            seen.add(norm(ref));r['id']=uid;r['split']='train'
            base=norm(r['input']['asr_top1']);gold=norm(ref)
            kws={norm(x['text']) for x in r['input']['hotwords'] if len(norm(x['text']))>=2}
            positive={k for k in kws if k in gold};preserved={k for k in positive if k in base}
            r['tags']={'preserve':bool(preserved),'repair':bool(positive-preserved),'distractor':bool(kws-positive)}
            pool.append(r)
    pool.sort(key=lambda r:hashlib.sha256(('20260908:'+r['id']).encode()).hexdigest())
    selected=[];taken=set();categories=collections.Counter();lengths=[]
    def add(r,category):
        if r['id'] in taken:return False
        out=formatted(r,True,True)
        text=tokenizer.apply_chat_template(out['messages'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        length=len(tokenizer(text,add_special_tokens=False)['input_ids'])
        if length>2048:excluded['over_2048_tokens']+=1;taken.add(r['id']);return False
        selected.append(out);taken.add(r['id']);categories[category]+=1;lengths.append(length);return True
    for category,quota in [('repair',1500),('preserve',2500),('distractor',1000)]:
        for r in pool:
            if categories[category]>=quota:break
            if r['tags'][category]:add(r,category)
    for r in pool:
        if len(selected)>=5000:break
        add(r,'fill')
    assert len(selected)==5000, len(selected)
    selected.sort(key=lambda r:hashlib.sha256(('shuffle:'+r['id']).encode()).hexdigest())
    dev=[json.loads(x) for x in (OUT/'dev.evidence.jsonl').read_text().splitlines() if x.strip()]
    assert len(dev)==len(devmap)==1334
    devseen=set()
    for r in dev:
        uid,ref=devmap[int(r['id'])];assert norm(ref)==norm(r['reference']);assert uid not in devseen
        assert not r['input'].get('oracle_hotwords');devseen.add(uid);r['id']=uid;r['split']='dev'
    assert not {r['id'] for r in selected}&devseen
    write('train.unified.jsonl',selected)
    for enabled,name in [(False,'old'),(True,'unified')]:
        formatted_dev=[formatted(r,enabled) for r in dev]
        write(f'dev.{name}.jsonl',formatted_dev)
        write(f'dev.screen.{name}.jsonl',formatted_dev[::4])
    dump('data_manifest.json',dict(train_rows=5000,dev_rows=1334,screen_rows=len(dev[::4]),
         categories=dict(categories),excluded=dict(excluded),max_train_tokens=max(lengths),
         mean_train_tokens=sum(lengths)/len(lengths),source_sha256=sha(source),
         dev_evidence_sha256=sha(OUT/'dev.evidence.jsonl'),bridge_sha256=sha(Path(bridge.__file__)),
         model=str(MODEL),starting_adapter=str(ADAPTER),starting_adapter_sha256=sha(ADAPTER/'adapter_model.safetensors'),
         split_audit='utterance IDs and normalized references checked against official dev/test; deduplicated train references',
         scope='small continued-LoRA pilot; contextual vocabulary follows existing upstream dataset protocol; not open-vocabulary claim',
         formatting='fixed original bridge vs same bridge with B v2 supplement; no gold prompt injection',
         selection='two saved checkpoints on fixed quarter-dev screen, both on full contextual dev; prefer designated recall>=0.90 then lowest CER, otherwise highest recall then lowest CER; failure to reach 0.90 explicitly reported',
         input_hashes={p.name:sha(p) for p in OUT.glob('*.jsonl')}))
    print('DATA_READY',json.dumps(dict(categories=categories,excluded=excluded,max_tokens=max(lengths))),flush=True)


if __name__=='__main__':main()
