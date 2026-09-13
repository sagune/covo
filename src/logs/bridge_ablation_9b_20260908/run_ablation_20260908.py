"""Locked 9B A/B/B+ diagnostic on 808 cached AISHELL-NE inputs, no training."""
import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
MIG = ROOT / 'cbwhisper_covo_migration_20260609_tar_extracted'
COVO = MIG / 'covo'
SOURCE = ROOT / 'src/logs/cb_sensevoice_acoustic_phrase_w14_covo_predictions_dpo30_aishell808_20260724.jsonl'
MODEL = MIG / 'models/Qwen3.5-9B'
ADAPTER = COVO / 'outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(path):
    sys.path.insert(0, str(COVO / 'src'))
    from covo.text import normalize_chinese_text
    from covo.metrics import edit_distance
    from opencc import OpenCC
    cc = OpenCC('t2s')
    def norm(s):
        return normalize_chinese_text(cc.convert(s))
    counts = collections.Counter()
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        ref, base, pred = (norm(x) for x in (r['reference'], r['input']['asr_top1'], r['prediction']))
        assert ref
        bd, pd = edit_distance(list(base), list(ref)), edit_distance(list(pred), list(ref))
        counts.update(samples=1, reference_chars=len(ref), baseline_edits=bd, prediction_edits=pd,
                      improved=int(pd<bd), worsened=int(pd>bd), unchanged=int(pd==bd),
                      baseline_exact=int(bd==0), prediction_exact=int(pd==0),
                      parse_warning_samples=int(bool(r.get('parse_warnings'))))
        kws = {norm(x['mention']) for x in r['input'].get('keyword_mentions', [])} - {''}
        for kw in kws:
            bh, ph = kw in base, kw in pred
            counts.update(mention_total=1, mention_base_hits=int(bh), mention_pred_hits=int(ph),
                          mention_lost=int(bh and not ph), mention_gained=int(ph and not bh))
        hotwords = {norm(x['text']) for x in r['input'].get('hotwords', [])} - {''}
        for kw in hotwords:
            if kw not in ref:
                counts['retrieved_nonreference_present'] += int(kw in pred)
                counts['retrieved_nonreference_newly_inserted'] += int(kw in pred and kw not in base)
    return dict(counts, cer=counts['prediction_edits']/counts['reference_chars'],
                baseline_cer=counts['baseline_edits']/counts['reference_chars'],
                normalization='OpenCC t2s + project NFKC/whitespace/punctuation removal',
                hotword_metric='unique keyword_mentions per utterance; NOT Recall@400',
                false_insertion_metric='retrieved word absent from reference, exact-substring proxy')


def main():
    started = time.time()
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Run directory already used; do not overwrite')
    for path in [SOURCE, MODEL/'config.json', ADAPTER/'adapter_model.safetensors']:
        assert path.is_file(), path
    v1path = ROOT/'src/analysis/bridge_b_20260907/cbsensevoice_covo_bridge.py'
    v2path = ROOT/'src/analysis/bridge_b_prompt_v2_20260907/cbsensevoice_covo_bridge.py'
    v1, v2 = load_module('bridge_v1', v1path), load_module('bridge_v2', v2path)
    records = [json.loads(x) for x in SOURCE.read_text(encoding='utf-8').splitlines()]
    assert len(records)==808 and len({str(r['id']) for r in records})==808
    manifest = dict(classification='post-hoc test diagnostic; fixed existing dev-selected checkpoint; no selection on this run',
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    model=str(MODEL), adapter=str(ADAPTER), adapter_sha256=sha(ADAPTER/'adapter_model.safetensors'),
                    source=str(SOURCE), source_sha256=sha(SOURCE), samples=808,
                    splits=dict(collections.Counter(r.get('split','') for r in records)),
                    variants={'A':'exact cached system/user prompt', 'B':'B v1 omission supplement', 'Bplus':'B v2 supplement + usage instructions'},
                    batch_size=8, max_new_tokens=128, temperature=0, disable_thinking=True,
                    bridge_v1_sha256=sha(v1path), bridge_v2_sha256=sha(v2path),
                    inference_script_sha256=sha(COVO/'scripts/infer_lora_text.py'),
                    stop_conditions='missing/corrupt input, inference nonzero exit/OOM, incomplete output; no automatic parameter changes')
    for variant, module in [('A', None), ('B', v1), ('Bplus', v2)]:
        with (OUT/f'{variant}.messages.jsonl').open('x',encoding='utf-8') as f:
            for r in records:
                out = {k:copy.deepcopy(r[k]) for k in ['id','source','dataset','split','reference','input']}
                messages = copy.deepcopy([m for m in r['messages'] if m['role'] in ['system','user']])
                users = [m for m in messages if m['role']=='user']
                assert len(users)==1
                if module:
                    lines = users[0]['content'].split('\n')
                    extra, audit = module.supplement_missing_kws_hotwords(r['input'], lines)
                    if extra:
                        users[0]['content']='\n'.join(lines[:-1]+extra+lines[-1:])
                    assert not audit['missing_after']
                    out['hotword_transmission']=audit
                out['messages']=messages
                assert out['input']==r['input']
                assert all(m['role']!='assistant' for m in messages)
                f.write(json.dumps(out,ensure_ascii=False)+'\n')
        manifest[f'{variant}_input_sha256']=sha(OUT/f'{variant}.messages.jsonl')
    dump(OUT/'manifest.json',manifest)
    results={}
    for variant in ['A','B','Bplus']:
        print('START',variant,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        phase=time.time()
        command=[sys.executable,str(COVO/'scripts/infer_lora_text.py'),
                 '--input',str(OUT/f'{variant}.messages.jsonl'),'--output',str(OUT/f'{variant}.predictions.jsonl'),
                 '--model-name-or-path',str(MODEL),'--adapter-path',str(ADAPTER),
                 '--batch-size','8','--max-new-tokens','128','--temperature','0','--progress-every','32','--disable-thinking']
        dump(OUT/f'{variant}.command.json',command)
        with (OUT/f'{variant}.inference.log').open('x',encoding='utf-8') as log:
            subprocess.run(command,cwd=COVO,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
        metrics=evaluate(OUT/f'{variant}.predictions.jsonl')
        assert metrics['samples']==808
        metrics['inference_and_eval_seconds']=time.time()-phase
        results[variant]=metrics
        dump(OUT/f'{variant}.metrics.json',metrics)
        dump(OUT/'results.json',results)
        print('DONE',variant,json.dumps(metrics,ensure_ascii=True),flush=True)
    dump(OUT/'ALL_DONE.json',dict(elapsed_seconds=time.time()-started,results=results))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        dump(OUT/'FAILED.json',dict(error=repr(error),time=time.time()))
        raise
