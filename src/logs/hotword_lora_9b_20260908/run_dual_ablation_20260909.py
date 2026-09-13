"""Full dev exploratory ablation, frozen checkpoint, serial GPU usage."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path('/root/autodl-tmp')
BASE=ROOT/'src/logs/hotword_lora_9b_20260908'
OUT=BASE/'dual_ablation_20260909'
MIG=ROOT/'cbwhisper_covo_migration_20260609_tar_extracted'
sys.path.insert(0,str(BASE))
from workflow import evaluate


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def command(name,args,cwd=ROOT):
    save('STATUS.json',dict(stage=name))
    save(name+'.command.json',args)
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:
        subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(exist_ok=False)
    rows=[json.loads(s) for s in (BASE/'dev.unified.jsonl').read_text().splitlines() if s.strip()]
    wavs={p.stem:str(p) for p in (ROOT/'datasets/aishell/data_aishell/wav/dev').rglob('*.wav')}
    assert len(rows)==1334 and all(r['id'] in wavs for r in rows)
    with (OUT/'audio_manifest.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(dict(id=r['id'],wav=wavs[r['id']],reference=r['reference'],split='dev'),ensure_ascii=False)+'\n')
    save('PROTOCOL.json',dict(classification='Exploratory full-dev; not held-out',samples=1334,
         adapter=str(BASE/'scheduled/high_lr3e-5/adapter/checkpoint-625'),
         neutral='Independent unadapted SenseVoice, no hotwords; not a pure decoding-only causal ablation',
         objective='Recall increase with no corpus CER increase vs existing high_lr3e-5 full_625 output',
         stop='Missing audio/ID mismatch, subprocess failure, or incomplete output stops the queue',
         workload='One 1334-audio neutral decode plus three 1334-sample 9B inference passes; no training'))
    command('neutral',[sys.executable,str(ROOT/'src/analysis/generate_sensevoice_chinesehp.py'),
         '--input',str(OUT/'audio_manifest.jsonl'),'--output',str(OUT/'neutral_raw.jsonl'),
         '--dataset','aishell','--beam-size','24','--token-topk','32','--max-nbest','1'])
    neutral=[json.loads(s) for s in (OUT/'neutral_raw.jsonl').read_text().splitlines() if s.strip()]
    assert len(neutral)==len(rows)
    with (OUT/'neutral.jsonl').open('x',encoding='utf-8') as f:
        for r in neutral:
            text=r['nbest'][0]
            f.write(json.dumps(dict(id=r['id'],prediction=text),ensure_ascii=False)+'\n')
    command('prepare',[sys.executable,str(BASE/'dual_evidence_20260909.py'),'--input',str(BASE/'dev.unified.jsonl'),
           '--neutral',str(OUT/'neutral.jsonl'),'--output-dir',str(OUT/'inputs')])
    results={'existing_best':evaluate(BASE/'scheduled/high_lr3e-5/full_625.predictions.jsonl')}
    for mode in ['cb_only','dual','delta']:
        prediction=OUT/(mode+'.predictions.jsonl')
        command(mode,[sys.executable,str(MIG/'covo/scripts/infer_lora_text.py'),
            '--input',str(OUT/'inputs'/(mode+'.jsonl')),'--output',str(prediction),
            '--model-name-or-path',str(MIG/'models/Qwen3.5-9B'),
            '--adapter-path',str(BASE/'scheduled/high_lr3e-5/adapter/checkpoint-625'),
            '--batch-size','8','--max-new-tokens','128','--temperature','0','--disable-thinking','--progress-every','128'],MIG/'covo')
        results[mode]=evaluate(prediction)
        assert results[mode]['samples']==1334
        save('RESULTS.json',results)
    for mode in ['cb_only','dual','delta']:
        results[mode]['joint_objective_met']=results[mode]['cer']<=results['existing_best']['cer'] and results[mode]['recall']>results['existing_best']['recall']
    save('RESULTS.json',results)
    save('DONE.json',dict(status='complete',classification='dev diagnostic'))
    save('STATUS.json',dict(stage='complete'))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save('FAILED.json',dict(error=repr(e)))
        raise
