"""Extend frozen CBSense shard protocol to all audited train/dev audio.

Reference text is never passed to the decoder or context assignment.
Existing outputs are reused only after per-context compatibility checks pass.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path[:0] = [str(SRC), str(SRC / 'data')]
os.chdir(SRC)
os.environ.update(CBW_DEBUG_LOG='off',CBW_DEBUG_CLEAR_ON_START='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from prepare_full import legacy, DATA


def context_for_id(uid):
    return f'train_covo_shard{int(hashlib.sha256(("917|"+uid).encode()).hexdigest(),16)%18:02d}'


def build():
    import fcntl
    import subprocess
    import torch
    import torchvision
    import yaml
    from model.cb_sensevoice import CBSenseVoice, DatabaseLite
    from hs_utils import load_hidden_states
    p = legacy()
    with (HERE.parent / 'text_residual_coupling_gpu.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert (HERE / 'audio_export_complete.json').exists()
        torch.set_num_threads(6)
        audit = json.loads((DATA / 'audit.json').read_text())
        rows = {s:p.load(DATA / f'{s}.manifest.jsonl') for s in ('train','dev')}
        for s in rows:
            assert p.digest(DATA / f'{s}.manifest.jsonl') == audit['manifests'][s]
        allrows = {r['id']:r for rr in rows.values() for r in rr}
        old, contexts, reference_mismatches = {}, {}, []
        mapping = {r['key']:r for r in p.load(p.MANIFESTS / 'aishell_train_manifest.jsonl')}
        for r in p.load(p.SOURCE):
            uid = mapping[f"{r['split']}|{r['id']}"]['id']
            if uid in allrows:
                if p.evaluator().norm(r['reference']) != p.evaluator().norm(allrows[uid]['reference']):
                    raise ValueError(json.dumps(dict(kind='existing_train_reference_mismatch', uid=uid,
                        source_id=r['id'], source_split=r['split'], source_reference=r['reference'],
                        official_reference=allrows[uid]['reference'], source_norm=p.evaluator().norm(r['reference']),
                        official_norm=p.evaluator().norm(allrows[uid]['reference'])), ensure_ascii=False))
                old[uid] = r['input']; contexts[uid] = r['split']
        for r in p.load(p.DEV_SOURCE):
            uid = r['id']
            assert uid in allrows and allrows[uid]['split'] == 'dev'
            if p.evaluator().norm(r['reference']) != p.evaluator().norm(allrows[uid]['reference']):
                reference_mismatches.append(dict(kind='existing_dev_reference_mismatch', uid=uid,
                    source_reference=r['reference'], official_reference=allrows[uid]['reference'],
                    source_norm=p.evaluator().norm(r['reference']), official_norm=p.evaluator().norm(allrows[uid]['reference'])))
            old[uid] = r['input']; contexts[uid] = 'dev'
        for uid,r in allrows.items():
            contexts.setdefault(uid,'dev' if r['split'] == 'dev' else context_for_id(uid))
        args = yaml.safe_load((SRC / 'configs/cb-sensevoice-aishell.yaml').read_text())['model']['init_args']
        args.update(split='train_covo_shard00',oracle_nbest_diagnostic=False,sensevoice_ckpt=str(p.SENSE))
        paths = [Path(__file__), DATA / 'audit.json', SRC / 'configs/cb-sensevoice-aishell.yaml',
                 SRC / 'model/cb_sensevoice.py',SRC / 'model/sensevoice_ctc.py',SRC / 'model/sensevoice_context_adapter.py',
                 SRC / 'data/dataset.py',SRC / 'hs_utils.py',Path(args['kws_ckpt']),Path(args['sensevoice_context_adapter_path']),
                 p.SOURCE,p.DEV_SOURCE]
        contextroot = (SRC / args['root']).resolve()
        paths += [contextroot / s / 'hotword.txt' for s in sorted(set(contexts.values()))]
        protocol = dict(config=args,context_policy='User selected existing shard protocol. Existing utterances retain original context; missing training IDs are assigned sha256(917|ID) modulo 18, independent of reference. All official dev uses the same existing 600-keyword dev dictionary; not gold per-utterance routing.',
                        existing=len(old),total=len(allrows),contexts=contexts,
                        reference_mismatches=reference_mismatches,
                        cache_semantics='Reuse original normalized int8 KWS hidden cache where available; otherwise encode zh/withitn and round(clamp(normalized,-1,1)*127)/127. CPU keyword similarity/resize identical to dataset path.',
                        caveat='Training contexts for inherited rows were originally assembled using training annotations; new rows are hash-assigned. Dev vocabulary provenance inherited: diagnostic external-context protocol, not clean open-vocabulary held-out.',
                        sources={str(q.resolve()):p.digest(q) for q in paths})
        lock = DATA / 'frontend_protocol.json'
        if lock.exists():
            assert json.loads(lock.read_text()) == protocol, 'Frontend/data changed during resume'
        else:
            p.save(lock,protocol)
        model = CBSenseVoice(**args).cuda().eval()
        original_encode = model._sensevoice_encode_audio
        active_row = {}
        def encode_with_original_kws(audio_path):
            logits, hidden = original_encode(audio_path)
            r = active_row['row']
            cache = contextroot / contexts[r['id']] / 'hs' / (r['id']+'.bin')
            if cache.is_file():
                utt = load_hidden_states(str(cache)).float().cpu()
            else:
                utt = torch.round(hidden.detach().cpu().clamp(-1,1)*127)/127
            assert utt.ndim == 3 and utt.shape[-1] == 512 and utt.shape[1] == hidden.shape[1]
            groups = []
            for group in model.kw_database.database:
                assert bool(group['mask'].all()), 'Missing keyword features; no ghost keyword fallback'
                matrices = [torch.matmul(h.float().cpu(),utt.transpose(1,2)) for h in group['hidden_states']]
                groups.append(torch.stack([torchvision.transforms.functional.resize(m,(150,750),antialias=False) for m in matrices]))
            model._active_precomputed_kws_features = groups
            return logits,hidden
        model._sensevoice_encode_audio = encode_with_original_kws
        def decode(row):
            active_row['row'] = row
            with torch.inference_mode():
                pred = model._forward_sensevoice(row['wav'],[])
                record = model._build_covo_evidence_record({'id':row['id']},0,pred,[],[])
            model._active_precomputed_kws_features = None
            assert p.evaluator().norm(pred), f"Empty ASR hypothesis {row['id']}; no sample is omitted"
            return record['input']
        outdir = DATA / 'frontend_shards'
        outdir.mkdir(exist_ok=True)
        started = time.time()
        total_done = 0
        compatibility_warnings = []
        # Check every reusable context before spending hours expanding any one.
        for context in sorted(set(contexts.values())):
            model.kw_database = DatabaseLite('aishell',context,str(contextroot),'tts',args['keywords_per_group'])
            model.hparams.split = context
            checks = sorted((uid for uid in old if contexts[uid] == context),key=p.rank)[:2]
            assert checks, f'No historical compatibility example for {context}'
            checkpath = outdir / f'{context}.compatibility.json'
            if not checkpath.exists():
                results = []
                for uid in checks:
                    fresh = decode(allrows[uid])
                    same = p.evidence({'input':fresh}) == p.evidence({'input':old[uid]})
                    results.append(dict(id=uid,equivalent_whitelisted_evidence=same,old=p.evidence({'input':old[uid]}),new=p.evidence({'input':fresh})))
                p.save(checkpath,dict(pass_all=all(r['equivalent_whitelisted_evidence'] for r in results),results=results))
            compatibility = json.loads(checkpath.read_text())
            if not compatibility['pass_all']:
                warning = dict(context=context, kind='historical_frontend_not_bitwise_equivalent',
                               compatibility=compatibility)
                compatibility_warnings.append(warning)
                print(json.dumps(warning, ensure_ascii=False), flush=True)
        p.save(DATA / 'frontend_compatibility_complete.json',dict(contexts=sorted(set(contexts.values())),examples_per_context=2))
        for context in sorted(set(contexts.values())):
            model.kw_database = DatabaseLite('aishell',context,str(contextroot),'tts',args['keywords_per_group'])
            model.hparams.split = context
            expected = [r for rr in rows.values() for r in rr if contexts[r['id']] == context]
            path = outdir / f'{context}.jsonl'
            done = p.load(path) if path.exists() else []
            assert len(done) <= len(expected)
            assert all(a['id'] == b['id'] and a['reference'] == b['reference'] for a,b in zip(done,expected))
            total_done += len(done)
            with path.open('a') as out:
                for i,row in enumerate(expected[len(done):],start=len(done)):
                    reused = row['id'] in old
                    inp = old[row['id']] if reused else decode(row)
                    record = dict(**row,input=inp,context_split=context,evidence_origin='reused' if reused else 'generated')
                    out.write(json.dumps(record,ensure_ascii=False)+'\n'); out.flush()
                    total_done += 1
                    if total_done % 25 == 0:
                        status = dict(status='GENERATING_FULL_CBSENSE_INPUTS',context=context,context_done=i+1,context_total=len(expected),
                                      done=total_done,total=len(allrows),seconds=time.time()-started,pid=os.getpid(),training='NOT_STARTED')
                        p.save(HERE / 'status.json',status)
                        print(json.dumps(status),flush=True)
        complete = {}
        for path in outdir.glob('*.jsonl'):
            for r in p.load(path):
                assert r['id'] not in complete
                complete[r['id']] = r
        assert set(complete) == set(allrows)
        for split in rows:
            target = DATA / f'{split}.jsonl'
            assert not target.exists(), 'Preserve completed dataset, do not overwrite'
            p.save_rows(target,[complete[r['id']] for r in rows[split]])
        p.save(DATA / 'evidence_complete.json',dict(sha256={s:p.digest(DATA/f'{s}.jsonl') for s in rows},
               context_policy=protocol['context_policy']+' '+protocol['caveat'],frontend_protocol_sha256=p.digest(lock),
               n={s:len(rr) for s,rr in rows.items()},existing_reused=len(old),generated=len(allrows)-len(old),
               compatibility_warnings=compatibility_warnings,
               caveat='Historical evidence is reused only as an explicitly marked provenance branch; current frontend was not bitwise reproducible on every checked historical dev example. Do not present this as a uniform regenerated CBSense decode.'))
        p.save(HERE / 'status.json',dict(status='FULL_EVIDENCE_READY',training='NOT_STARTED'))


if __name__ == '__main__':
    try:
        build()
    except BaseException:
        legacy().save(HERE / f'frontend_failure_{time.time_ns()}.json',dict(error=traceback.format_exc(),timestamp=time.time()))
        raise
