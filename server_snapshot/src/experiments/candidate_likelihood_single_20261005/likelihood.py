"""Frozen candidate conditional likelihood: one target per forward, no padding."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / 'acoustic_rereview_20261005'
if not PREV.exists():
    PREV = HERE.parent / 'acoustic-rereview-20261005'
EVID = Path('/root/autodl-tmp/src/experiments/acoustic_review_20261005')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()


def previous():
    sys.path.insert(0, str(PREV))
    import rereview
    return rereview


def reordered(q, order):
    assert order in ('original_order', 'reversed_order')
    qq = copy.deepcopy(q)
    if order == 'reversed_order': qq['A'], qq['B'] = qq['B'], qq['A']
    return qq


def target(q, key):
    assert key in ('A', 'B')
    return json.dumps(dict(text=q[key]['sentence']), ensure_ascii=False)


def target_positions(target_length, retained_logits):
    # A causal logit at position t predicts token t+1. All targets end at the batch right edge.
    assert target_length > 0 and retained_logits >= target_length + 1
    return list(range(retained_logits - target_length - 1, retained_logits - 1))


def choose(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    return 'A' if a >= b else 'B'  # exact numeric tie: A, fixed before scoring


def single_layout(encoded):
    ids = encoded['ids']; t = len(encoded['target_ids'])
    assert t > 0 and encoded['prompt_tokens'] > 0
    assert len(ids) == encoded['prompt_tokens'] + t and ids[-t:] == encoded['target_ids']
    return dict(ids=[ids], mask=[[1] * len(ids)], positions=[list(range(len(ids)))], retained=t+1)


def compose(draft, pending):
    out = draft
    for q, role in reversed(pending):
        lo, hi = q['output_span']
        modified = next(q[k] for k in ('A', 'B') if q[k]['role'] == 'modified')
        original = next(q[k] for k in ('A', 'B') if q[k]['role'] == 'original')
        assert out[lo:hi] == modified['text']
        if role == 'original': out = out[:lo] + original['text'] + out[hi:]
    return out


def prepare():
    assert not (HERE / 'protocol.json').exists()
    r = previous(); pc = load(PREV / 'protocol.json')
    seal = load(PREV / 'decision_seal.json')
    assert sha(PREV / 'protocol.json') == seal['protocol_sha256']
    for path, expected in pc['sources'].items(): assert sha(path) == expected, path
    qs = rows(PREV / 'queries.jsonl')
    assert len(qs) == len({q['edit_id'] for q in qs}) == 190
    assert len({q['id'] for q in qs}) == 180
    for q in qs:
        assert set(q) == {'id', 'edit_id', 'output_span', 'draft', 'input_evidence', 'A', 'B', 'evidence'}
        assert not {'reference', 'keyword_mentions', 'oracle_hotwords'} & set(q['input_evidence'])
        assert len(q['A']['text']) == len(q['B']['text']) and q['A']['sentence'] != q['B']['sentence']
    # Copy identical bytes, no new candidate routing and no reference access for construction.
    with (HERE / 'queries.jsonl').open('xb') as stream: stream.write((PREV / 'queries.jsonl').read_bytes())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc['base'], trust_remote_code=True)
    lengths = {}; target_lengths = []
    conditions = []
    for order in ('original_order', 'reversed_order'):
        for mode in pc['modes']:
            name = mode + '__' + order
            conditions.append(dict(name=name, mode=mode, order=order))
            ll = []
            for source_q in qs:
                q = reordered(source_q, order)
                prompt = tok.apply_chat_template(r.messages(q, mode), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(prompt, add_special_tokens=False)['input_ids']
                for key in ('A', 'B'):
                    t = tok(target(q, key), add_special_tokens=False)['input_ids']
                    assert len(ids) + len(t) <= 4096 and t
                    ll.append(len(ids) + len(t)); target_lengths.append(len(t))
            lengths[name] = dict(min=min(ll), max=max(ll), mean=sum(ll) / len(ll))
    deps = {**pc['sources'], str(PREV / 'protocol.json'): sha(PREV / 'protocol.json'),
            str(PREV / 'decision_seal.json'): sha(PREV / 'decision_seal.json')}
    for name in ('likelihood.py', 'test_likelihood.py', 'audit_download.py', 'PLAN.md', 'queries.jsonl'):
        deps[str(HERE / name)] = sha(HERE / name)
    failed = HERE.parent / 'candidate_likelihood_20261005' / 'protocol.json'
    deps[str(failed)] = sha(failed)
    c = dict(experiment='Frozen9B full candidate conditional likelihood: single target, no padding',
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), sources=deps, base=pc['base'], checkpoint=pc['checkpoint'],
        baseline=pc['baseline'], dev=pc['dev'], baseline_replay_ids=pc['baseline_replay_ids'],
        n=1334, chars=21102, mentions=599, queries=190, reviewed_utterances=180,
        conditions=conditions, batch_sequences=1, chunk_queries=1, no_padding=True, max_sequence=4096, seed=917, dtype='bfloat16',
        no_training=True, no_connector=True, sequence_lengths=lengths, target_tokens_minmax=[min(target_lengths), max(target_lengths)],
        prompt='Exactly prior rereview.messages; reversed condition only swaps complete A/B candidate objects',
        target='json.dumps({text: full_candidate_sentence},ensure_ascii=False), standard spaces; separately tokenized continuation; no EOS/end-of-turn token',
        score='Sum FP32 next-token log probabilities for every target token including JSON syntax, excluding all prompt and padding positions; no token/character length normalization in primary',
        likelihood_logits='One target per forward with inputs_embeds; all-one attention_mask, position_ids=0..L-1, no padding, use_cache=False; retain last target_tokens+1 logits',
        selection='Larger full conditional log likelihood; exact numeric tie selects A; no abstention target, no score threshold or recovery',
        primary_order='original_order locked; reversed_order is robustness control, not selected or averaged by dev results',
        routing=pc['routing'], normalization='NFKC/remove whitespace/Unicode punctuation; no OpenCC',
        primary='Full1334 corpusCER/hotword recall vs oldB/text and none; true vs zero/mismatch, paired95CI, harmful/helpful edits after score sealing',
        stopping='Nonfinite score/hash/identity/position alignment/same-path repeat/OOM failure: stop, preserve all output; no protocol reduction or automatic retry',
        numerical_audit='First4 input-order pairs: repeat identical single no-padding path in reverse job order, exact token-logp equality; first target trimmed/full-logits discrepancy<=0.005 nat; no cross-padding/batch equality requirement',
        user_authorized_change='2026-10-05: 那就无padding呗，继续; previous failed protocol preserved, not amended',
        limitations=[*pc['limitations'],
            'Forced binary selection eliminates free generation failures but also removes explicit abstention; not a same-task accuracy estimate',
            'Autoregressive full target probability combines language/orthography/tokenization with evidence; not acoustic-only confidence',
            'Equal candidate character length does not ensure equal tokenizer length; token-count asymmetry recorded, mean logp diagnostic only',
            'Full-target probability contains downstream context effects; not a pure target-span classifier',
            'Candidate scoring of the same repeated dev is diagnostic; no trained model or unseen test claim',
            'Cross-batch/padding numerical disagreement remains documented in prior audits; all current targets use one fixed unpadded single path',
            'No process-local padding patch is needed or loaded: all attention-mask elements equal one'])
    save(HERE / 'protocol.json', c); save(HERE / 'status.json', dict(status='PREPARED', queries=190, conditions=8))
    print(json.dumps(dict(status='PREPARED', sequences=3040, lengths=lengths, target_tokens_minmax=c['target_tokens_minmax'])), flush=True)


def config():
    c = load(HERE / 'protocol.json')
    for path, expected in c['sources'].items(): assert sha(path) == expected, path
    return c


class Scorer:
    def __init__(self, c):
        self.c = c; self.r = previous(); self.rt = self.r.Runtime(c)

    def encode(self, jobs):
        tok = self.rt.tok; encoded = []
        for q, mode, key in jobs:
            prompt = tok.apply_chat_template(self.r.messages(q, mode), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            a = tok(prompt, add_special_tokens=False)['input_ids']
            b = tok(target(q, key), add_special_tokens=False)['input_ids']
            assert len(a) + len(b) <= self.c['max_sequence']
            encoded.append(dict(ids=a + b, target_ids=b, prompt_tokens=len(a)))
        return encoded

    def score(self, jobs, all_logits=False):
        import torch
        result = []
        with torch.inference_mode():
            for x in self.encode(jobs):
                layout = single_layout(x)
                ids = torch.tensor(layout['ids'], device='cuda'); mask = torch.tensor(layout['mask'], device='cuda')
                pos = torch.tensor(layout['positions'], device='cuda')
                assert ids.shape[0] == 1 and torch.all(mask == 1)
                logits = self.rt.model(inputs_embeds=self.rt.model.get_input_embeddings()(ids), attention_mask=mask,
                    position_ids=pos, use_cache=False, logits_to_keep=0 if all_logits else layout['retained']).logits
                t = len(x['target_ids'])
                offsets = target_positions(t, logits.shape[1])
                chosen = logits[0, offsets].float()
                wanted = torch.tensor(x['target_ids'], device='cuda')
                lp = (chosen.gather(1, wanted[:, None]).squeeze(1) - torch.logsumexp(chosen, dim=-1)).cpu().tolist()
                assert len(lp) == t and all(math.isfinite(v) for v in lp)
                result.append(dict(logp=math.fsum(lp), token_logp=lp, target_ids=x['target_ids'],
                    target_tokens=t, prompt_tokens=x['prompt_tokens'], mean_logp=math.fsum(lp) / t,
                    model_batch_size=1, padding_tokens=0, sequence_tokens=len(x['ids'])))
        return result


def run():
    c = config(); r = previous(); p = r.w.legacy_module(HERE.parent / 'staged_coupling_20260917')
    qs = rows(HERE / 'queries.jsonl')
    with r.w.gpu_guard(p):
        scorer = Scorer(c); replay = []
        # Historical replay verifies checkpoint identity; reference fields are never rendered.
        old = {x['id']: x for x in rows(c['baseline'])}
        for uid in c['baseline_replay_ids']:
            x = p.generate(scorer.rt.model, scorer.rt.tok, None, old[uid], 'B', 'text')
            assert x['model_output'] == old[uid]['model_output'], uid
            replay.append(dict(id=uid, raw_exact=True))
        jobs = [(q, 'none', key) for q in qs[:4] for key in ('A', 'B')]
        singles = scorer.score(jobs)
        repeats = list(reversed(scorer.score(list(reversed(jobs)))))
        assert singles == repeats, 'Identical single no-padding scoring not repeatable'
        # Verify causal target positions against full logits on the first input only.
        full = scorer.score(jobs[:1], all_logits=True)[0]
        assert max(abs(a-b) for a,b in zip(full['token_logp'], singles[0]['token_logp'])) <= .005
        save(HERE / 'runtime_audit.json', dict(status='PASS', adapter_reload_exact=True, baseline_replay=replay,
            single_no_padding_repeat_exact=True, repeat_reverse_job_order=True, model_batch_size=1, padding_tokens=0,
            causal_trimmed_vs_full_logits=True, trainable_parameters=0, connector_instantiated=False,
            first8_scores=singles, first8_repeat_scores=repeats, reference_not_rendered=True))
        began = time.time()
        for cc in c['conditions']:
            name = cc['name']; final = HERE / (name + '.scores.jsonl'); partial = Path(str(final) + '.partial')
            assert not final.exists() and not partial.exists()
            with partial.open('x', encoding='utf-8') as stream:
                for start in range(0, len(qs), c['chunk_queries']):
                    chunk = [reordered(q, cc['order']) for q in qs[start:start+c['chunk_queries']]]
                    jobs = [(q, cc['mode'], key) for q in chunk for key in ('A', 'B')]
                    score = scorer.score(jobs)
                    for j, q in enumerate(chunk):
                        candidates = {key: score[2*j+i] for i, key in enumerate(('A', 'B'))}
                        key = choose(candidates['A']['logp'], candidates['B']['logp'])
                        rec = dict(id=q['id'], edit_id=q['edit_id'], condition=name,
                            candidate_roles={k:q[k]['role'] for k in ('A','B')}, candidates=candidates,
                            logp_A_minus_B=candidates['A']['logp']-candidates['B']['logp'],
                            exact_tie=candidates['A']['logp']==candidates['B']['logp'],
                            selected_key=key, selected_role=q[key]['role'])
                        stream.write(json.dumps(rec, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
                    rec = dict(status='SCORING', condition=name, done=start+len(chunk), total=190, seconds=time.time()-began)
                    save(HERE / 'status.json', rec)
                    if rec['done'] % 20 == 0 or rec['done'] == 190: print(json.dumps(rec), flush=True)
            assert [x['edit_id'] for x in rows(partial)] == [q['edit_id'] for q in qs]
            partial.rename(final)
    save(HERE / 'score_seal.json', dict(protocol_sha256=sha(HERE/'protocol.json'), reference_used=False,
        hashes={cc['name']:sha(HERE/(cc['name']+'.scores.jsonl')) for cc in c['conditions']}))
    save(HERE / 'status.json', dict(status='SCORES_COMPLETE_NOT_EVALUATED'))


def report():
    c=config(); seal=load(HERE/'score_seal.json'); assert not seal['reference_used']
    assert seal['protocol_sha256']==sha(HERE/'protocol.json')
    r=previous(); p=r.w.legacy_module(HERE.parent/'staged_coupling_20260917')
    qs=rows(HERE/'queries.jsonl'); old=rows(c['baseline']); dev=rows(c['dev'])
    labels={v['edit_id']:v for v in load(EVID/'posthoc_cases.json')}
    baseline=p.metrics(old, protected=False); assert (baseline['summary']['errors'],baseline['summary']['hits'])==(622,469)
    metrics={}; decisions={}; records={}; prediction_hashes={}
    for cc in c['conditions']:
        name=cc['name']; assert sha(HERE/(name+'.scores.jsonl'))==seal['hashes'][name]
        ss=rows(HERE/(name+'.scores.jsonl')); records[name]=ss
        assert len(ss)==190 and [x['edit_id'] for x in ss]==[q['edit_id'] for q in qs]
        by={}; cells={k:dict(n=0,original_selected=0,modified_selected=0) for k in ('harmful','helpful','neutral')}
        for q, s in zip(qs,ss):
            by.setdefault(q['id'],[]).append((q,s['selected_role']))
            cell=cells[labels[q['edit_id']]['diagnosis']]; cell['n']+=1; cell[s['selected_role']+'_selected']+=1
        predictions=[]
        changes=dict(changed=0,better=0,worse=0,equal=0,removed_errors=0,added_errors=0)
        for b,e in zip(old,dev):
            assert b['id']==e['id'] and b['reference']==e['reference']
            pred=compose(r.evidence.norm(b['prediction']),by[b['id']]) if b['id'] in by else b['prediction']
            predictions.append(dict(id=b['id'],reference=b['reference'],input=b['input'],prediction=pred,parsed=True,
                condition=name,reviewed_edits=len(by.get(b['id'],[]))))
            x=r.evidence.norm(pred); y=r.evidence.norm(b['prediction']); ref=r.evidence.norm(b['reference'])
            if x!=y:
                a=r.evidence.align(ref,x)[0]; bb=r.evidence.align(ref,y)[0]
                changes['changed']+=1; changes['better']+=a<bb; changes['worse']+=a>bb; changes['equal']+=a==bb
                changes['removed_errors']+=max(0,bb-a); changes['added_errors']+=max(0,a-bb)
        assert len(predictions)==1334 and [x['id'] for x in predictions]==[x['id'] for x in old]
        with (HERE/(name+'.predictions.jsonl')).open('x',encoding='utf-8') as stream:
            for x in predictions:stream.write(json.dumps(x,ensure_ascii=False)+'\n')
        mm=p.metrics(predictions,protected=False); mm['diagnostics']=r.diagnose(predictions,p.evaluator().norm)
        assert (mm['summary']['n'],mm['summary']['chars'],mm['summary']['mentions'])==(1334,21102,599)
        assert mm['summary']['parse_failures']==0
        save(HERE/(name+'.metrics.json'),mm); metrics[name]=mm
        decisions[name]=dict(cells=cells,utterance_changes=changes,exact_ties=sum(x['exact_tie'] for x in ss),
            unequal_target_token_counts=sum(x['candidates']['A']['target_tokens']!=x['candidates']['B']['target_tokens'] for x in ss))
        prediction_hashes[name]=sha(HERE/(name+'.predictions.jsonl'))
    contrasts={}
    for order in ('original_order','reversed_order'):
        a='true__'+order
        contrasts[order]={m:p.bootstrap(metrics[a]['per_utterance'],metrics[m+'__'+order]['per_utterance']) for m in ('none','zero','mismatch')}
    agreement={m:sum(x['selected_role']==y['selected_role'] for x,y in zip(records[m+'__original_order'],records[m+'__reversed_order'])) for m in ('none','true','zero','mismatch')}
    result=dict(status='COMPLETE',baseline=baseline['summary'],
        summaries={n:v['summary'] for n,v in metrics.items()},diagnostics={n:v['diagnostics'] for n,v in metrics.items()},
        decisions=decisions,paired95ci_vs_old={n:p.bootstrap(v['per_utterance'],baseline['per_utterance']) for n,v in metrics.items()},
        true_minus_controls95ci=contrasts,order_role_agreement_of190=agreement,
        protocol_sha256=sha(HERE/'protocol.json'),score_hashes=seal['hashes'],prediction_hashes=prediction_hashes,
        limitations=c['limitations'],no_training=True,no_connector=True,no_threshold_or_restore=True)
    save(HERE/'summary.json',result); save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','run','report','pipeline']); a=ap.parse_args()
    try:
        if a.command=='pipeline': prepare(); run(); report()
        else: globals()[a.command]()
    except BaseException:
        failure=dict(error=traceback.format_exc(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        save(HERE/('failure_'+str(time.time_ns())+'.json'),failure); save(HERE/'status.json',dict(status='FAILED',**failure)); raise
