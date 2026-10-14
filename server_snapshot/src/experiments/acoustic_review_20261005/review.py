"""Frozen native SenseVoice evidence for old-9B edits; no output restoration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import unicodedata

HERE = Path(__file__).resolve().parent
PARENT = Path('/root/autodl-tmp/src/experiments/joint_lora_20260923')
SENSE = Path('/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(text)))
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P')).strip()


def phones(text):
    if not text or not all('\u4e00' <= ch <= '\u9fff' for ch in text):
        return None
    from pypinyin import lazy_pinyin, Style
    result = lazy_pinyin(text, style=Style.NORMAL)
    if any(not x.isascii() or not x.isalpha() for x in result):
        return None
    return result


def align(a, b):
    """Input/output-only edit alignment, diagonal/deletion/insertion ties."""
    n, m = len(a), len(b)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (a[i-1] != b[j-1]), dp[i-1][j]+1, dp[i][j-1]+1)
    i, j = n, m
    ops = []
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            ops.append(('M' if a[i-1] == b[j-1] else 'S', i-1, j-1, a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('D', i-1, j, a[i-1], '')); i -= 1
        else:
            ops.append(('I', i, j-1, '', b[j-1])); j -= 1
    return dp[n][m], list(reversed(ops))


def candidates(uid, original, output):
    """This function has no reference/annotation input."""
    a, b = norm(original), norm(output)
    _, ops = align(a, b)
    pa, pb = phones(a), phones(b)
    blocks = []
    pending = []
    for op in ops + [('M', len(a), len(b), '', '')]:
        if op[0] != 'M':
            pending.append(op)
            continue
        if not pending:
            continue
        start_a, start_b = pending[0][1:3]
        aa = ''.join(x[3] for x in pending)
        bb = ''.join(x[4] for x in pending)
        end_a, end_b = start_a + len(aa), start_b + len(bb)
        reverted = b[:start_b] + aa + b[end_b:]
        q = dict(edit_id=uid + ':' + str(len(blocks)), input_span=[start_a,end_a],
                 output_span=[start_b,end_b], original=aa, modified=bb,
                 operations=[x[0] for x in pending], candidate_original_context=reverted,
                 candidate_modified_context=b)
        if pa is None or pb is None:
            q['kind'] = 'unsupported_pronunciation'
        elif all(x[0] == 'S' for x in pending):
            q['original_phones'] = pa[start_a:end_a]
            q['modified_phones'] = pb[start_b:end_b]
            q['kind'] = 'different_phone_substitution' if q['original_phones'] != q['modified_phones'] else 'same_phone_substitution'
        else:
            q['kind'] = 'insertion_deletion_or_mixed'
        q['original_context_phones'] = phones(reverted)
        q['modified_context_phones'] = pb
        blocks.append(q); pending = []
    return dict(id=uid, normalized_input=a, normalized_output=b, edits=blocks)


def prepare():
    assert not (HERE/'protocol.json').exists()
    parent = load(PARENT/'protocol.json')
    dev, old = rows(parent['dev']), rows(PARENT/'B/dev/text.predictions.jsonl')
    assert len(dev) == len(old) == 1334
    assert len({r['id'] for r in dev}) == 1334
    cache = load(parent['cache_manifest'])
    manifest = []
    for r, b in zip(dev, old):
        assert (r['id'],r['reference'],r['input']) == (b['id'],b['reference'],b['input'])
        assert r['split'] == 'dev' and b['condition'] == 'text' and b['parsed']
        q = candidates(r['id'], r['input']['asr_top1'], b['prediction'])
        # Changing the reference does not change the inference manifest.
        assert q == candidates(r['id'], b['input']['asr_top1'], b['prediction'])
        manifest.append(q)
    assert sum(len(norm(r['reference'])) for r in dev) == 21102
    assert sum(align(norm(r['reference']),norm(b['prediction']))[0] for r,b in zip(dev,old)) == 622
    assert sum(align(norm(r['reference']),norm(r['input']['asr_top1']))[0] for r in dev) == 804
    with (HERE/'input_edits.jsonl').open('x',encoding='utf-8') as stream:
        for q in manifest:
            stream.write(json.dumps(q,ensure_ascii=False)+'\n')
    dependencies = [Path(__file__), HERE/'test_review.py', HERE/'PLAN.md', HERE/'input_edits.jsonl',
                    PARENT/'protocol.json', Path(parent['dev']), Path(parent['cache_manifest']),
                    PARENT/'B/dev/text.predictions.jsonl', PARENT/'B/dev/text.metrics.json',
                    SENSE/'model.pt', SENSE/'config.yaml', SENSE/'tokens.json',
                    SENSE/'chn_jpn_yue_eng_ko_spectok.bpe.model']
    kinds = {}
    for q in manifest:
        for e in q['edits']:
            kinds[e['kind']] = kinds.get(e['kind'],0)+1
    cache_ids = set(r['id'] for r in dev) | {parent['audio_permutation'][r['id']] for r in dev}
    cache_records = {uid:cache[uid] for uid in sorted(cache_ids)}
    for uid,info in cache_records.items():
        assert sha(info['path']) == info['sha256'], uid
    protocol = dict(experiment='Native SenseVoice acoustic review of frozen old-best 9B edits',
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), dev=parent['dev'],
        old_predictions=str(PARENT/'B/dev/text.predictions.jsonl'),
        baseline='Old B/text final2162: 622/21102 CER, 469/599 mentions; no new 9B generation',
        old_checkpoint=load(PARENT/'B/selected_checkpoint.json'),
        sources={str(p):sha(p) for p in dependencies}, cache_records=cache_records,
        audio_permutation={r['id']:parent['audio_permutation'][r['id']] for r in dev},
        modes=['true','zero','mismatch'], n=1334, chars=21102, mentions=599, edit_counts=kinds,
        trainable='NONE; native pretrained SenseVoice CTC layer only; no probe/connector/9B updates',
        primary='Native character CTC sum log-likelihood difference modified-minus-original-context for input/output-only different-phone substitution blocks',
        auxiliary='Fixed dictionary phonetic marginal of native single-Han token posteriors, conditional on Han-or-blank; same context paired CTC. No learned mapping.',
        controls='Cached raw encoder features true/erased-zero/inherited utterance mismatch. Zero is NOT encoded silence.',
        scoring='No reference, mentions, hotwords or known-correct flag used for candidate scoring. No thresholds optimized.',
        diagnostic='Reference read ONLY after all three evidence files sealed; negative error delta on single-edit reversion labels harmful, positive labels helpful; neutral reported separately',
        output_policy='NO output restoration, no protected CER or deployed correction claim; old predictions remain unchanged',
        stopping='Any cache/source/ID/finite-score fault stops; unsupported candidates explicitly retained with reasons, never silently dropped',
        limitations=['Repeated historical contextual1334 dev, not official14326 full dev or new heldout',
          'Native character CTC includes orthographic/context bias, not pure pronunciation or calibrated confidence',
          'Toneless dictionary pronunciation is not waveform-verified; tone/polyphone/same-phone spelling limitations',
          'Phonetic token aggregation changes repeat-collapse semantics and excludes non-Han tokens; exploratory fixed auxiliary',
          'Single-edit same-context whole-utterance CTC marginal avoids reference boundaries but is not an independent local audio judge',
          'Existing float16 cached encoder frames; no hotword bonus in these acoustic scores',
          'Detecting damaging edits is not proof that 9B can use evidence or that end-to-end CER improves'])
    save(HERE/'protocol.json',protocol)
    save(HERE/'status.json',dict(status='PREPARED',n=1334,edit_counts=kinds))
    print(json.dumps(dict(status='PREPARED',edit_counts=kinds)),flush=True)


def configuration():
    c = load(HERE/'protocol.json')
    for path,expected in c['sources'].items():
        assert sha(path) == expected, path
    return c


def ctc_score(lp, ids, blank=0):
    import torch
    if any(x is None for x in ids):
        return dict(score=None,reason='unsupported_native_token_or_phone')
    if len(lp) < len(ids) + sum(x==y for x,y in zip(ids,ids[1:])):
        return dict(score=None,reason='insufficient_ctc_frames')
    if not ids:
        return dict(score=float(lp[:,blank].sum()),reason=None)
    loss = torch.nn.functional.ctc_loss(lp[:,None,:].contiguous(),
        torch.tensor(ids,device=lp.device,dtype=torch.long),torch.tensor([len(lp)]),torch.tensor([len(ids)]),
        blank=blank,reduction='sum',zero_infinity=False)
    score = -float(loss)
    assert math.isfinite(score)
    return dict(score=score,reason=None)


def score():
    import fcntl
    import numpy as np
    import torch
    from funasr import AutoModel
    c = configuration(); manifest = rows(HERE/'input_edits.jsonl')
    assert len(manifest) == 1334
    with (HERE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Share the repository's guard with all coupling experiments.
        sys.path.insert(0,str(PARENT.parent/'text_residual_coupling_20260920'))
        import workflow as w
        p = w.legacy_module(PARENT.parent/'staged_coupling_20260917')
        with w.gpu_guard(p), torch.inference_mode():
            torch.set_num_threads(6)
            wrapper = AutoModel(model=str(SENSE),device='cuda:0',disable_update=True)
            model = wrapper.model.eval(); tokenizer = wrapper.kwargs['tokenizer']
            assert int(model.blank_id) == 0
            tokens = load(SENSE/'tokens.json'); assert len(tokens) == model.ctc.ctc_lo.out_features
            # No train/dev references are used to build this vocabulary.
            groups = {}
            for i,token in enumerate(tokens):
                if len(token)==1 and '\u4e00'<=token<='\u9fff':
                    ph = phones(token)
                    if ph is not None:
                        groups.setdefault(ph[0],[]).append(i)
            pv = ['<blank>']+sorted(groups); pi = {x:i for i,x in enumerate(pv)}
            groups_idx = [torch.tensor(groups[x],device='cuda',dtype=torch.long) for x in pv[1:]]
            save(HERE/'native_interface.json',dict(blank=0,native_vocab=len(tokens),phone_vocab=pv,
                mapping=groups,ctc_source_file=__import__('inspect').getfile(type(model.ctc)),
                native_layer_sha256=sha(SENSE/'model.pt')))
            began = time.time()
            for mode in c['modes']:
                output = HERE/(mode+'.evidence.jsonl')
                assert not output.exists(), 'Preserve existing evidence; do not overwrite'
                partial = Path(str(output)+'.partial'); assert not partial.exists()
                with partial.open('x',encoding='utf-8') as stream:
                    for i,q in enumerate(manifest):
                        uid = c['audio_permutation'][q['id']] if mode=='mismatch' else q['id']
                        info = c['cache_records'][uid]
                        assert sha(info['path']) == info['sha256']
                        frames = torch.from_numpy(np.load(info['path'],allow_pickle=False).copy()).float().cuda()
                        assert tuple(frames.shape)==(info['frames'],512) and torch.isfinite(frames).all()
                        if mode=='zero': frames.zero_()
                        lp = model.ctc.log_softmax(frames[None,:,:])[0]
                        assert torch.isfinite(lp).all()
                        # Sum native single-character posteriors sharing a dictionary pronunciation.
                        plp = torch.stack([lp[:,0]]+[torch.logsumexp(lp[:,g],dim=-1) for g in groups_idx],dim=-1)
                        plp = plp-torch.logsumexp(plp,dim=-1,keepdim=True)
                        cache_scores = {}
                        def scores_for(text,ph):
                            key = (text,tuple(ph) if ph is not None else None)
                            if key not in cache_scores:
                                tids = [int(x) for x in tokenizer.encode(text)]
                                bad = any(x==0 or x in range(1,16) or x>=25000 for x in tids)
                                # No filtering, dropping or substituted unknown targets.
                                native = dict(score=None,reason='unknown_or_special_native_token') if bad else ctc_score(lp,tids)
                                phone = dict(score=None,reason='non_Han_or_unknown_pronunciation') if ph is None else ctc_score(plp,[pi.get(x) for x in ph])
                                cache_scores[key] = dict(native=native,phonetic_marginal=phone,native_ids=tids)
                            return cache_scores[key]
                        result = dict(id=q['id'],mode=mode,audio_id=uid,edits=[])
                        for e in q['edits']:
                            a = scores_for(e['candidate_original_context'],e['original_context_phones'])
                            b = scores_for(e['candidate_modified_context'],e['modified_context_phones'])
                            z = dict(edit_id=e['edit_id'],kind=e['kind'],scores={})
                            for method in ('native','phonetic_marginal'):
                                aa,bb = a[method],b[method]
                                z['scores'][method] = dict(original=aa,modified=bb,
                                    modified_minus_original=(bb['score']-aa['score']) if aa['score'] is not None and bb['score'] is not None else None)
                            result['edits'].append(z)
                        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                        if (i+1)%100==0 or i+1==len(manifest):
                            rec = dict(status='SCORING',mode=mode,done=i+1,total=len(manifest),seconds=time.time()-began)
                            save(HERE/'status.json',rec); print(json.dumps(rec),flush=True)
                rr = rows(partial)
                assert [r['id'] for r in rr] == [r['id'] for r in manifest]
                partial.rename(output)
            save(HERE/'evidence_seal.json',dict(protocol_sha256=sha(HERE/'protocol.json'),
                manifest_sha256=sha(HERE/'input_edits.jsonl'),
                evidence_sha256={m:sha(HERE/(m+'.evidence.jsonl')) for m in c['modes']},
                reference_used=False,completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    save(HERE/'status.json',dict(status='EVIDENCE_COMPLETE_NOT_DIAGNOSED'))


def interval(per, fn, count=2000):
    rng = random.Random(917)
    values = []
    for _ in range(count):
        sampled = rng.choices(per,k=len(per))
        value = fn(sampled)
        if value is not None: values.append(value)
    values.sort()
    return [values[int(.025*len(values))],values[min(len(values)-1,int(.975*len(values)))]] if values else None


def report():
    c = configuration(); seal = load(HERE/'evidence_seal.json')
    assert seal['protocol_sha256'] == sha(HERE/'protocol.json') and seal['reference_used'] is False
    for m in c['modes']: assert sha(HERE/(m+'.evidence.jsonl')) == seal['evidence_sha256'][m]
    manifest = rows(HERE/'input_edits.jsonl')
    dev, baseline = rows(c['dev']),rows(c['old_predictions'])
    evid = {m:rows(HERE/(m+'.evidence.jsonl')) for m in c['modes']}
    assert all([r['id'] for r in rr]==[r['id'] for r in dev] for rr in [manifest,baseline,*evid.values()])
    # Read evaluator after freezing all evidence. No code or parameter selection.
    sys.path.insert(0,str(PARENT.parent/'text_residual_coupling_20260920'))
    import workflow as w
    p = w.legacy_module(PARENT.parent/'staged_coupling_20260917')
    evaluator = p.evaluator()
    assert all(norm(r['reference'])==evaluator.norm(r['reference']) for r in dev)
    bm = p.metrics(baseline,protected=False)
    assert (bm['summary']['n'],bm['summary']['chars'],bm['summary']['mentions'],bm['summary']['errors'],bm['summary']['hits'])==(1334,21102,599,622,469)
    full_cases = []
    per = []
    for i,(r,q,b) in enumerate(zip(dev,manifest,baseline)):
        ref = norm(r['reference']); out = q['normalized_output']; inp = q['normalized_input']
        errs = align(ref,out)[0]
        item = dict(id=r['id'],cases=[])
        for j,e in enumerate(q['edits']):
            revert_errs = align(ref,e['candidate_original_context'])[0]
            delta = revert_errs-errs
            z = dict(e,reference=ref,original_input_correct=inp==ref,output_errors=errs,
                     single_reversion_errors=revert_errs,reversion_error_delta=delta,
                     diagnosis='harmful' if delta<0 else 'helpful' if delta>0 else 'neutral',conditions={})
            for m in c['modes']:
                ee = evid[m][i]['edits'][j]
                assert ee['edit_id']==e['edit_id'] and ee['kind']==e['kind']
                z['conditions'][m]=ee['scores']
            item['cases'].append(z); full_cases.append(z)
        per.append(item)
    summaries = {}
    for method in ('native','phonetic_marginal'):
        summaries[method] = {}
        for mode in c['modes']:
            cells = {}
            for kind in ('different_phone_substitution','same_phone_substitution','insertion_deletion_or_mixed','unsupported_pronunciation'):
                chosen = [z for z in full_cases if z['kind']==kind]
                groups = {}
                for label in ('harmful','helpful','neutral'):
                    selected = [z for z in chosen if z['diagnosis']==label]
                    valid = [z for z in selected if z['conditions'][mode][method]['modified_minus_original'] is not None]
                    negative = [z for z in valid if z['conditions'][mode][method]['modified_minus_original']<0]
                    positive = [z for z in valid if z['conditions'][mode][method]['modified_minus_original']>0]
                    groups[label]=dict(n=len(selected),scored=len(valid),original_preferred=len(negative),modified_preferred=len(positive),ties=len(valid)-len(negative)-len(positive),
                        original_preferred_fraction=len(negative)/len(valid) if valid else None,
                        error_units=sum(abs(z['reversion_error_delta']) for z in valid),
                        original_preferred_error_units=sum(abs(z['reversion_error_delta']) for z in negative))
                cells[kind]=groups
            summaries[method][mode]=cells
        def discrimination(items,mode):
            harm_total=harm_correct=help_total=help_correct=0
            for item in items:
                for z in item['cases']:
                    if z['kind']!='different_phone_substitution':continue
                    gap=z['conditions'][mode][method]['modified_minus_original']
                    if gap is None:continue
                    if z['diagnosis']=='harmful': harm_total+=1;harm_correct+=gap<0
                    elif z['diagnosis']=='helpful':help_total+=1;help_correct+=gap>0
            return 50*(harm_correct/harm_total+help_correct/help_total) if harm_total and help_total else None
        summaries[method]['balanced_accuracy_pp']={m:discrimination(per,m) for m in c['modes']}
        summaries[method]['balanced_accuracy_95ci_pp']={m:interval(per,lambda items,m=m:discrimination(items,m)) for m in c['modes']}
        summaries[method]['true_minus_control_balanced_accuracy_95ci_pp']={m:interval(per,lambda items,m=m:discrimination(items,'true')-discrimination(items,m)) for m in ('zero','mismatch')}
    correct_input = {}
    for method in ('native','phonetic_marginal'):
        correct_input[method] = {}
        for mode in c['modes']:
            chosen=[z for z in full_cases if z['kind']=='different_phone_substitution' and z['original_input_correct']]
            valid=[z for z in chosen if z['conditions'][mode][method]['modified_minus_original'] is not None]
            rejected=[z for z in valid if z['conditions'][mode][method]['modified_minus_original']<0]
            correct_input[method][mode]=dict(n=len(chosen),scored=len(valid),original_preferred=len(rejected),
                original_preferred_fraction=len(rejected)/len(valid) if valid else None,
                recoverable_error_units=sum(-z['reversion_error_delta'] for z in valid),
                original_preferred_recoverable_error_units=sum(-z['reversion_error_delta'] for z in rejected))
    save(HERE/'posthoc_cases.json',full_cases)
    save(HERE/'summary.json',dict(status='COMPLETE_DIAGNOSTIC',baseline=bm['summary'],
         edit_counts=c['edit_counts'],methods=summaries,correct_input_subgroup=correct_input,
         protocol_sha256=sha(HERE/'protocol.json'),
         evidence_seal=seal,posthoc_cases_sha256=sha(HERE/'posthoc_cases.json'),
         limitations=c['limitations'],no_output_changed=True))
    save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(load(HERE/'summary.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','score','report','pipeline'])
    args=parser.parse_args()
    try:
        if args.command=='pipeline':
            prepare();score();report()
        else: globals()[args.command]()
    except BaseException:
        save(HERE/('failure_'+str(time.time_ns())+'.json'),dict(command=args.command,error=traceback.format_exc()))
        save(HERE/'status.json',dict(status='FAILED',command=args.command,error=traceback.format_exc()))
        raise
