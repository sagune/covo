"""Post-run independent standard-library integrity and edit-distance audit."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def distance(ref, hyp):
    previous = list(range(len(hyp)+1))
    for i, token in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(previous[j]+1, current[j-1]+1, previous[j-1]+(token != other)))
        previous = current
    return previous[-1]

def main():
    protocol = read(HERE/'protocol.json')
    summary = read(HERE/'summary.json')
    assert read(HERE/'status.json')['status'] == 'ALL_DONE'
    assert sha(HERE/'protocol.json') == summary['protocol_sha256']
    for path, expected in protocol['sources'].items():
        assert sha(Path(path)) == expected, path
    expected = rows(HERE/'dev_targets.jsonl')
    assert len(expected) == 1334
    assert sum(len(r['syllables']) for r in expected) == 21102
    audit = dict(status='PASS',protocol_sha256=summary['protocol_sha256'],heads={})
    selected_paths = []
    for head in protocol['heads']:
        directory = HERE/head
        selected = read(directory/'selected_checkpoint.json')
        checkpoint = Path(selected['checkpoint'])
        metadata = read(checkpoint/'metadata.json')
        assert selected['epoch'] == metadata['epoch'] == 10
        assert metadata['updates'] == 5410
        assert metadata['protocol_sha256'] == summary['protocol_sha256']
        assert sha(checkpoint/'state.pt') == metadata['state_sha256']
        selected_paths += [checkpoint/'state.pt',checkpoint/'metadata.json']
        conditions = {}
        for mode in protocol['modes']:
            path = directory/'dev'/(mode+'.predictions.jsonl')
            predictions = rows(path)
            assert len(predictions) == len(expected)
            total = 0
            for prediction, reference in zip(predictions,expected):
                assert prediction['id'] == reference['id']
                assert prediction['reference'] == reference['reference']
                assert prediction['reference_syllables'] == reference['syllables']
                assert prediction['mode'] == mode
                audio_id = protocol['audio_permutation'][reference['id']] if mode == 'mismatch' else reference['id']
                assert prediction['audio_id'] == audio_id
                total += distance(reference['syllables'],prediction['predicted_syllables'])
            reported = summary['heads'][head]['summaries'][mode]
            assert (reported['n'],reported['syllables'],reported['errors']) == (1334,21102,total)
            assert abs(reported['syer'] - total/21102) < 1e-12
            assert sha(path) == summary['heads'][head]['prediction_hashes'][mode]
            assert reported['S']+reported['D']+reported['I'] == total
            conditions[mode] = dict(n=len(predictions),reference_syllables=21102,errors=total,syer=total/21102,sha256=sha(path))
        audit['heads'][head] = dict(epoch=10,updates=5410,checkpoint_sha256=sha(checkpoint/'state.pt'),conditions=conditions)
    (HERE/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    target = HERE/'phonetic_probe_audited_results.zip'
    assert not target.exists(), 'Do not overwrite existing archive'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*')):
            if not path.is_file() or path == target or '__pycache__' in path.parts:
                continue
            if 'checkpoints' in path.parts and path not in selected_paths:
                continue
            if path.suffix == '.lock':
                continue
            archive.write(path,path.relative_to(HERE))
    print(json.dumps(audit))
    print('archive_sha256='+sha(target))

if __name__ == '__main__':
    main()
