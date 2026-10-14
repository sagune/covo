"""Post-run completeness, independent edit-count check and paired comparisons."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'validation_Afull_3243_full_dev_20260917'
spec=importlib.util.spec_from_file_location('pilot',HERE/'run.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    ev=p.evaluator()
    expected=p.load(HERE/'run/dev.jsonl')
    config=json.loads((OUT/'protocol.json').read_text())
    mapping=config['audio_permutation']
    files=sorted(OUT.glob('*.predictions.jsonl'))
    assert len(files)==8 and not list(OUT.glob('*.partial'))
    results={}
    for path in files:
        rows=p.load(path)
        p.unique(rows,'id')
        assert len(rows)==1334
        assert [r['id'] for r in rows]==[r['id'] for r in expected]
        assert [r['reference'] for r in rows]==[r['reference'] for r in expected]
        assert [r['input'] for r in rows]==[r['input'] for r in expected]
        for row in rows:
            assert row['audio_id']==(mapping[row['id']] if row['condition']=='mismatch' else row['id'])
        independent_errors=sum(distance(ev.norm(r['reference']),ev.norm(r['prediction'])) for r in rows)
        prefix=path.name.removesuffix('.predictions.jsonl')
        metric=json.loads((OUT/f'{prefix}.metrics.json').read_text())
        recomputed=p.metrics(rows)
        assert recomputed['summary']==metric['summary']
        assert recomputed['per_utterance']==metric['per_utterance']
        assert metric['summary']['errors']==independent_errors
        assert metric['summary']['chars']==21102 and metric['summary']['mentions']==599
        if prefix.startswith(('text_','B_probe_')):
            protected=json.loads((OUT/f'{prefix}.protected.metrics.json').read_text())
            assert p.metrics(rows,True)==protected
        results[prefix]=dict(status='PASS',sha256=p.digest(path),**metric['summary'])

    def comparison(name_a,name_b):
        a=json.loads((OUT/f'{name_a}.metrics.json').read_text())
        b=json.loads((OUT/f'{name_b}.metrics.json').read_text())
        av,bv=a['per_utterance'],b['per_utterance']
        return dict(a=name_a,b=name_b,
            error_difference=a['summary']['errors']-b['summary']['errors'],
            cer_difference_pp=100*(a['summary']['cer']-b['summary']['cer']),
            hit_difference=a['summary']['hits']-b['summary']['hits'],
            recall_difference_pp=100*(a['summary']['recall']-b['summary']['recall']),
            improved_vs_comparator=sum(x['errors']<y['errors'] for x,y in zip(av,bv)),
            worsened_vs_comparator=sum(x['errors']>y['errors'] for x,y in zip(av,bv)),
            **p.bootstrap(av,bv))
    comparisons={
        'audio_only_new_vs_old':comparison('A_true','A_before_true'),
        'raw_audio_vs_raw_text':comparison('B_probe_true','text_text'),
        'protected_audio_vs_protected_text':comparison('B_probe_true.protected','text_text.protected'),
        'correct_vs_zero_audio_raw':comparison('B_probe_true','B_probe_zero'),
        'correct_vs_mismatched_audio_raw':comparison('B_probe_true','B_probe_mismatch')}
    audit=dict(status='PASS',n_groups=8,n_each=1334,chars_each=21102,mentions_each=599,
               total_predictions=10672,groups=results,comparisons=comparisons,
               audit_code_sha256=p.digest(__file__),source_summary_sha256=p.digest(OUT/'summary.json'))
    p.save(OUT/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
