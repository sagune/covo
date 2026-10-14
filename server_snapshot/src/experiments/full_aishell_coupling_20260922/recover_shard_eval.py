"""Finish the locked four-condition shard evaluation without changing training.

The original evaluator erroneously required off.jsonl despite a protocol with
only text/true/zero/mismatch. Keep original sources, checkpoints and predictions.
Audit existing complete predictions and run only missing conditions.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'text_residual_coupling_20260920'))
import workflow as w
from diagnostics import diagnose


def validate(rows, expected, mode, config):
    if len(rows) > len(expected):
        raise ValueError('Extra predictions')
    for r, e in zip(rows, expected):
        if (r['id'], r['reference'], r['input'], r['condition']) != (e['id'], e['reference'], e['input'], mode):
            raise ValueError(f"Prediction identity/evidence mismatch: {r['id']}")
        want_audio = config['audio_permutation'][e['id']] if mode == 'mismatch' else e['id']
        if r['audio_id'] != want_audio:
            raise ValueError(f"Wrong audio control: {r['id']}")


def main(architecture):
    directory = HERE / 'runs' / f'shard_{architecture}'
    c, p = w.configuration(directory)
    assert c['modes'] == ['text', 'true', 'zero', 'mismatch']
    checkpoint = directory / 'final'
    meta = w.read(checkpoint / 'metadata.json')
    selected = w.read(directory / 'selected_checkpoint.json')
    assert meta['complete'] and meta['step'] == c['steps'] == 2162
    assert meta['examples'] == c['train_n'] == 17294
    assert meta['architecture'] == architecture and meta['protocol_sha256'] == c['protocol_sha256']
    assert selected['connector_sha256'] == meta['connector_sha256'] == w.sha(checkpoint / 'connector.pt')
    rows = p.load(c['dev'])
    assert len(rows) == c['dev_n'] == 1334 and len({r['id'] for r in rows}) == len(rows)
    dest = directory / 'dev'
    dest.mkdir(exist_ok=True)
    recovery = dest / 'recovery_audit'
    recovery.mkdir(exist_ok=True)
    note = dict(reason='Original summary read off predictions absent from the locked four-condition protocol',
                code_sha256=w.sha(Path(__file__)), protocol_sha256=c['protocol_sha256'],
                checkpoint_sha256=meta['connector_sha256'], modes=c['modes'],
                off_validation='Not scheduled; no off predictions fabricated or inferred to pass',
                existing_outputs={f.name:w.sha(f) for f in dest.glob('*.predictions.jsonl')})
    w.save(recovery / f'attempt_{time.time_ns()}.json', note)
    missing = [m for m in c['modes'] if not (dest / f'{m}.predictions.jsonl').exists()]
    if missing:
        with w.gpu_guard(p):
            runtime = w.Runtime(c, p, checkpoint=checkpoint)
            for mode in missing:
                path = dest / f'{mode}.predictions.jsonl'
                partial = Path(str(path)+'.partial')
                prior = p.load(partial) if partial.exists() else []
                validate(prior, rows, mode, c)
                began = time.time()
                with partial.open('a', encoding='utf-8') as stream:
                    for i in range(len(prior), len(rows)):
                        row = rows[i]
                        pred = runtime.predict(row, mode, c['audio_permutation'][row['id']] if mode == 'mismatch' else None)
                        stream.write(json.dumps(pred, ensure_ascii=False)+'\n')
                        stream.flush()
                        if (i+1) % 25 == 0 or i+1 == len(rows):
                            status = dict(status='EVALUATING_SHARD_DEV', architecture=architecture, mode=mode,
                                          done=i+1, total=len(rows), modes=c['modes'], seconds=time.time()-began)
                            w.save(directory / 'status.json', status)
                            print(json.dumps(status), flush=True)
                validate(p.load(partial), rows, mode, c)
                assert len(p.load(partial)) == len(rows)
                partial.rename(path)
    metrics, predictions = {}, {}
    for mode in c['modes']:
        path = dest / f'{mode}.predictions.jsonl'
        rr = p.load(path)
        validate(rr, rows, mode, c)
        assert len(rr) == len(rows)
        metric = p.metrics(rr, protected=False)
        metric['error_diagnostics'] = diagnose(rr, p.evaluator().norm)
        assert metric['summary']['chars'] == 21102 and metric['summary']['mentions'] == 599
        assert sum(metric['error_diagnostics'].get(k, 0) for k in ('S','D','I')) == metric['summary']['errors']
        old = dest / f'{mode}.metrics.json'
        if old.exists():
            assert w.read(old) == metric, f'Existing metric differs from re-scoring: {mode}'
        else:
            w.save(old, metric)
        w.save(recovery / f'{mode}.metrics.json', metric)
        metrics[mode], predictions[mode] = metric, rr
    # The original input baseline is scored with exactly the same normalization and labels.
    original = [dict(r, prediction=r['input']['asr_top1'], parsed=True) for r in rows]
    input_metrics = p.metrics(original, protected=False)
    w.save(recovery / 'cbsense_input.metrics.json', input_metrics)
    norm = p.evaluator().norm
    true = predictions['true']
    summary = dict(label=c['label'], train_n=c['train_n'], dev_n=c['dev_n'],
                   summaries={k:v['summary'] for k,v in metrics.items()}, input_summary=input_metrics['summary'],
                   error_diagnostics={k:v['error_diagnostics'] for k,v in metrics.items()},
                   paired_bootstrap={m:p.bootstrap(metrics['true']['per_utterance'], metrics[m]['per_utterance'])
                                     for m in ('text','zero','mismatch')},
                   true_vs_control_prediction_changes={m:sum(norm(a['prediction']) != norm(b['prediction'])
                                                            for a,b in zip(true,predictions[m]))
                                                       for m in ('text','zero','mismatch')},
                   no_output_repairs=True, normalization=c['normalization'], data_caveat=c['data_caveat'],
                   context_caveat=c['external_context_caveat'],
                   initialization_caveat='Residual zero-initialized; MLP starts from historical Afull. Not equal pretraining histories.',
                   protocol_sha256=c['protocol_sha256'], checkpoint_sha256=meta['connector_sha256'],
                   recovery_note=note,
                   predictions_sha256={m:w.sha(dest/f'{m}.predictions.jsonl') for m in c['modes']})
    target = dest / 'summary.json'
    if target.exists():
        raise FileExistsError('Preserve existing final summary')
    w.save(target, summary)
    w.save(directory / 'status.json',dict(status='COMPLETED_SHARD_DEV',summary=str(target),
                                        completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    bundle = HERE / f'shard_{architecture}_audited_results.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for q in [*dest.rglob('*.json'), *dest.glob('*.predictions.jsonl'), directory/'protocol.json',
                  directory/'selected_checkpoint.json', directory/'final/metadata.json', directory/'status.json',
                  directory/'training_started.json',directory/'training.jsonl', Path(__file__)]:
            archive.write(q, str(q.relative_to(HERE)))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', choices=['residual','mlp'])
    args = parser.parse_args()
    try:
        main(args.architecture)
    except BaseException:
        folder = HERE / 'runs' / f'shard_{args.architecture}'
        w.save(folder / f'recovery_failure_{time.time_ns()}.json',dict(error=traceback.format_exc()))
        raise
