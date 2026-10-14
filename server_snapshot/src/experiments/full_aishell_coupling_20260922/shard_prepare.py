"""Prepare the existing ~17k CBSense shard evidence as an auditable train/dev run."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp')
LEGACY = HERE.parent / 'staged_coupling_20260917'
OUT = HERE / 'shard_data'


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def pmod():
    spec=importlib.util.spec_from_file_location('shard_legacy',LEGACY/'run.py')
    p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p); return p


def main():
    p=pmod(); ev=p.evaluator(); OUT.mkdir(exist_ok=True)
    trainmap={r['key']:r for r in p.load(p.MANIFESTS/'aishell_train_manifest.jsonl')}
    devmap={r['id']:r for r in p.load(p.MANIFESTS/'aishell_dev_manifest.jsonl')}
    dev=[]
    for r in p.load(p.DEV_SOURCE):
        m=devmap[r['id']]
        assert ev.norm(m['reference'])==ev.norm(r['reference'])
        dev.append(dict(id=r['id'],wav=m['wav'],reference=m['reference'],input=r['input'],split='dev',context_split='dev'))
    devids={r['id'] for r in dev}; devrefs={ev.norm(r['reference']) for r in dev}
    train=[]; duplicate_ref=[]
    for r in p.load(p.SOURCE):
        m=trainmap[f"{r['split']}|{r['id']}"]
        assert ev.norm(m['reference'])==ev.norm(r['reference'])
        if m['id'] in devids: raise ValueError('train/dev ID overlap')
        if ev.norm(m['reference']) in devrefs:
            duplicate_ref.append(m['id']); continue
        train.append(dict(id=m['id'],wav=m['wav'],reference=m['reference'],input=r['input'],split='train',context_split=r['split']))
    print(json.dumps(dict(train=len(train),dev=len(dev),duplicate_refs=len(duplicate_ref))),flush=True)
    assert len(dev)==1334
    for r in train+dev:
        cache=p.RUN/'audio'/f"{r['id']}.npy"
        assert cache.is_file() and cache.stat().st_size>0
        assert ev.norm(r['input']['asr_top1'])
    def rows(path,items):
        with path.open('w',encoding='utf-8') as f:
            for x in items: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    rows(OUT/'train.jsonl',train); rows(OUT/'dev.jsonl',dev)
    protocol=dict(version=1,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),label='AISHELL CBSense shard evidence training; diagnostic, not held-out',
        train_n=len(train),dev_n=len(dev),train=str(OUT/'train.jsonl'),dev=str(OUT/'dev.jsonl'),
        source_evidence_n=17301,excluded_duplicate_dev_references=len(duplicate_ref),
        normalization='restore_eval: NFKC, remove spaces/punctuation; NO OpenCC',
        train_policy=f'All 17301 existing CBSense evidence rows with {len(duplicate_ref)} normalized-reference duplicates of the fixed 1334 dev removed; no correctness filtering',
        context_policy='Existing train shard context retained; existing dev context retained',
        steps=math.ceil(len(train)/8) if False else 2157, accumulation=8, lr=5e-5, weight_decay=.01,warmup=32,
        max_sequence=4096,max_new_tokens=96,seed=917,width=256,heads=4,pool=4,
        modes=['text','true','zero','mismatch'],selection=f'Fixed final step after one complete pass over {len(train)} rows',
        caveat='Historical CBSense evidence is reused as produced; this is not a newly regenerated uniform frontend. Existing Afull initialization has inherited training history.',
        sources={str(q):sha(q) for q in [p.SOURCE,p.DEV_SOURCE,p.MANIFESTS/'aishell_train_manifest.jsonl',p.MANIFESTS/'aishell_dev_manifest.jsonl',p.ALIGNED,LEGACY/'run.py']},
        data_hashes={s:sha(OUT/f'{s}.jsonl') for s in ('train','dev')})
    import math
    protocol['steps']=math.ceil(len(train)/protocol['accumulation'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps(dict(status='PREPARED_SHARD',train_n=len(train),dev_n=len(dev)),indent=2),encoding='utf-8')
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
