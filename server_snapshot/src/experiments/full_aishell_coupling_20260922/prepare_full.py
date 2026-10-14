"""Full official AISHELL train/dev audit and missing acoustic cache export.

Never reads test audio or test predictions. No model training is launched here.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/root/autodl-tmp')
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
LEGACY = HERE.parent / 'staged_coupling_20260917'


def legacy():
    spec = importlib.util.spec_from_file_location('full_aishell_legacy', LEGACY / 'run.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    return p


def audit():
    p = legacy()
    norm = p.evaluator().norm
    source = ROOT / 'datasets/aishell/data_aishell'
    waves = {s: sorted((source / 'wav' / s).rglob('*.wav')) for s in ('train', 'dev')}
    ids = {s: {w.stem for w in ws} for s, ws in waves.items()}
    for s in ids:
        assert len(ids[s]) == len(waves[s]), f'Duplicate {s} wave IDs'
    assert not ids['train'] & ids['dev']
    wanted = ids['train'] | ids['dev']
    refs = {}
    transcript = source / 'transcript/aishell_transcript_v0.8.txt'
    for line in transcript.read_text().splitlines():
        uid, *words = line.split()
        if uid not in wanted:
            continue  # Test references are not used or collected.
        if uid in refs:
            raise ValueError(f'Duplicate official reference {uid}')
        ref = ''.join(words)
        assert norm(ref), uid
        refs[uid] = ref
    rows, exclusions = {}, []
    devrefs = {norm(refs[u]) for u in ids['dev'] if u in refs}
    for split, ws in waves.items():
        rows[split] = []
        for wav in ws:
            uid = wav.stem
            reason = None
            if uid not in refs:
                reason = 'official_transcript_missing'
            elif split == 'train' and norm(refs[uid]) in devrefs:
                reason = 'normalized_reference_also_in_complete_dev'
            if reason:
                exclusions.append(dict(id=uid, split=split, wav=str(wav), reason=reason))
                continue
            rows[split].append(dict(id=uid, split=split, wav=str(wav), reference=refs[uid]))
    assert len(rows['dev']) == 14326
    assert len(ids['train'] & refs.keys()) == 120098
    assert len(ids['dev'] & refs.keys()) == 14326
    oldtrain = p.load(LEGACY / 'overnight/full_train.jsonl')
    oldnorm = {r['id']: norm(r['reference']) for r in oldtrain}
    existing = {}
    mapping = {r['key']: r for r in p.load(p.MANIFESTS / 'aishell_train_manifest.jsonl')}
    for r in p.load(p.SOURCE):
        m = mapping[f"{r['split']}|{r['id']}"]
        assert norm(m['reference']) == norm(r['reference'])
        assert m['id'] not in existing
        existing[m['id']] = dict(source_split=r['split'], source_id=r['id'])
    DATA.mkdir(parents=True, exist_ok=False)
    for split in rows:
        p.save_rows(DATA / f'{split}.manifest.jsonl', rows[split])
    p.save_rows(DATA / 'exclusions.jsonl', exclusions)
    p.save(DATA / 'existing_evidence_index.json', existing)
    missing = {s: [r['id'] for r in rs if not (p.RUN / 'audio' / (r['id'] + '.npy')).is_file()]
               for s, rs in rows.items()}
    p.save(DATA / 'missing_audio_ids.json', missing)
    audit = dict(
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        label='Complete available official labelled AISHELL train/dev; dev previously examined in part; diagnostic, not test',
        raw_audio={s: len(v) for s, v in waves.items()},
        labelled_audio={s: len(ids[s] & refs.keys()) for s in ids},
        train_n=len(rows['train']), dev_n=len(rows['dev']),
        dev_chars=sum(len(norm(r['reference'])) for r in rows['dev']),
        exclusions={reason: sum(r['reason'] == reason for r in exclusions)
                    for reason in sorted({r['reason'] for r in exclusions})},
        missing_audio_cache={s: len(v) for s, v in missing.items()},
        existing_train_evidence=len(set(existing) & {r['id'] for r in rows['train']}),
        evidence_caveat='Existing 17301 outputs use shard-specific dictionaries; not interchangeable with a global 20000-keyword frontend.',
        historical_Afull_manifest_dev_reference_overlap=sum(v in devrefs for v in oldnorm.values()),
        historical_adapter_caveat='Existing checkpoint-625 and Afull weights have inherited training history; full-dev evaluation remains diagnostic. Warm-start leakage must be audited before a held-out claim.',
        normalization='restore_eval: NFKC, remove whitespace and punctuation; NO OpenCC',
        acoustic_cache='SenseVoiceSmall zh/woitn; valid encoder frames after four queries; unnormalized float16 512D',
        train_policy='All labelled train except normalized-reference overlap with complete dev; no error/correctness filtering',
        dev_policy='All 14326 labelled dev; 5 unlabelled audio listed separately, never silently omitted',
        sources={str(q): p.digest(q) for q in [Path(__file__), LEGACY / 'run.py', transcript,
                  ROOT / '.dsh_checks/restore_eval.py', p.SOURCE]},
        manifests={s: p.digest(DATA / f'{s}.manifest.jsonl') for s in rows},
    )
    p.save(DATA / 'audit.json', audit)
    p.save(HERE / 'status.json', dict(status='FULL_SPLITS_AUDITED', **{k: audit[k] for k in ('train_n','dev_n')}))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def export():
    import fcntl
    import numpy as np
    import torch
    from funasr import AutoModel
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
    import subprocess
    p = legacy()
    audit = json.loads((DATA / 'audit.json').read_text())
    rows = []
    for split in ('train', 'dev'):
        path = DATA / f'{split}.manifest.jsonl'
        assert p.digest(path) == audit['manifests'][split]
        rows.extend(p.load(path))
    with (HERE.parent / 'text_residual_coupling_gpu.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True)
        assert not active.strip(), f'GPU already busy: {active}'
        torch.set_num_threads(6)
        started = time.time()
        p.save(HERE / 'status.json', dict(status='EXPORTING_FULL_AUDIO', pid=os.getpid(), total=len(rows), started=started))
        wrapper = AutoModel(model=str(p.SENSE), device='cuda:0', disable_update=True)
        model, frontend = wrapper.model.eval(), wrapper.kwargs['frontend']
        cache = p.RUN / 'audio'
        exported = reused = 0
        with torch.inference_mode(), (HERE / 'audio_cache_audit.jsonl').open('a') as log:
            for i, row in enumerate(rows):
                path = cache / f"{row['id']}.npy"
                if path.exists():
                    value = np.load(path, mmap_mode='r', allow_pickle=False)
                    assert value.ndim == 2 and value.shape[1] == 512 and len(value) and np.isfinite(value).all(), row['id']
                    reused += 1
                else:
                    audio = load_audio_text_image_video(row['wav'], fs=frontend.fs, audio_fs=None, data_type='sound', tokenizer=None)
                    speech, lengths = extract_fbank(audio, data_type='sound', frontend=frontend)
                    speech, lengths = speech.cuda(), lengths.cuda()
                    valid = int(lengths[0])
                    def query(ids):
                        return model.embed(torch.tensor([ids], device='cuda', dtype=torch.long))
                    x = torch.cat([query([model.lid_dict['zh']]), query([1,2]), query([model.textnorm_dict['woitn']]), speech], dim=1)
                    enc, out_lengths = model.encoder(x, lengths + 4)
                    if isinstance(enc, tuple):
                        enc = enc[0]
                    assert int(out_lengths[0]) == valid + 4
                    value = enc[0, 4:valid+4].float().cpu().numpy().astype(np.float16)
                    assert value.shape == (valid,512) and np.isfinite(value).all()
                    temp = Path(str(path) + '.full20260922.tmp')
                    with temp.open('wb') as stream:
                        np.save(stream, value)
                    os.replace(temp, path)
                    exported += 1
                log.write(json.dumps(dict(id=row['id'], split=row['split'], path=str(path), frames=len(value), sha256=p.digest(path))) + '\n')
                if (i+1) % 100 == 0 or i+1 == len(rows):
                    log.flush()
                    status = dict(status='EXPORTING_FULL_AUDIO', done=i+1, total=len(rows), exported=exported, reused=reused,
                                  seconds=time.time()-started, pid=os.getpid())
                    p.save(HERE / 'status.json', status)
                    print(json.dumps(status), flush=True)
        p.save(HERE / 'audio_export_complete.json', dict(total=len(rows), exported=exported, reused=reused, seconds=time.time()-started))
        p.save(HERE / 'status.json', dict(status='FULL_AUDIO_READY_FRONTEND_PENDING', total=len(rows), training='NOT_STARTED'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit','export'])
    args = parser.parse_args()
    try:
        {'audit': audit, 'export': export}[args.command]()
    except BaseException:
        if HERE.exists():
            legacy().save(HERE / 'failure.json', dict(command=args.command, error=traceback.format_exc(), timestamp=time.time()))
        raise
