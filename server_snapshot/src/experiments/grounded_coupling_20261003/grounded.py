"""Train-only pronunciation contrast and paired text/audio correction learning.

No test mining, inference-time routing, transcription fallback, or new frontend.
The MLP architecture, prompt, initializer and schedule match the prior B arm.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'joint_lora_20260923'
sys.path.insert(0, str(OLD))
import experiment as old
w = old.w


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def pronunciation(text):
    """Conservative: non-Han strings are ineligible; disregard tones here."""
    if not text or not all('\u4e00' <= ch <= '\u9fff' for ch in text):
        return None
    from pypinyin import lazy_pinyin, Style
    return tuple(lazy_pinyin(text, style=Style.NORMAL))


def auxiliary_selected(uid):
    return int(hashlib.sha256(('917|' + uid).encode()).hexdigest()[:8], 16) % 4 == 0


def build_pair(row, p):
    assert row['split'] == 'train', 'Never mine dev/test negatives'
    norm = p.evaluator().norm
    ref = norm(row['reference'])
    ev = p.evidence(row)
    pool = list(dict.fromkeys([ev['asr_top1'], *ev['nbest']]))
    ref_pron = pronunciation(ref)
    choices = []
    rejected = {'same_phonemes': 0, 'non_han': 0, 'too_far': 0}
    for index, text in enumerate(pool):
        value = norm(text)
        distance = edit_distance(ref, value)
        if distance == 0:
            continue
        if distance > 4:
            rejected['too_far'] += 1
            continue
        pron = pronunciation(value)
        if ref_pron is None or pron is None:
            rejected['non_han'] += 1
            continue
        if pron == ref_pron:
            # Includes tone-only alternatives: no claim to resolve tones in v1.
            rejected['same_phonemes'] += 1
            continue
        choices.append((distance, index, text))
    negative = min(choices)[2] if choices else None
    return dict(id=row['id'], correct_input=ref == norm(ev['asr_top1']),
                negative=negative, contrast=bool(negative and auxiliary_selected(row['id'])),
                rejected=rejected)


def prepare():
    assert not (HERE / 'protocol.json').exists(), 'Immutable run already prepared'
    c, p = old.configuration()
    train, dev = p.load(c['train']), p.load(c['dev'])
    assert (len(train), len(dev)) == (17294, 1334)
    assert all(r['split'] == 'train' for r in train)
    assert all(r['split'] == 'dev' for r in dev)
    p.unique(train, 'id'); p.unique(dev, 'id')
    assert not ({r['id'] for r in train} & {r['id'] for r in dev})
    norm = p.evaluator().norm
    assert not ({norm(r['reference']) for r in train} & {norm(r['reference']) for r in dev})
    pairs = []
    for i, row in enumerate(train):
        pairs.append(build_pair(row, p))
        if (i + 1) % 2000 == 0:
            print(json.dumps(dict(prepared=i + 1, total=len(train))), flush=True)
    p.save_rows(HERE / 'train_pairs.jsonl', pairs)
    assert sum(x['correct_input'] for x in pairs) == 4254
    by_id = {r['id']: r for r in train}
    # Audit actual negative target lengths; positives already audited by prior run.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p.BASE, trust_remote_code=True)
    cache = w.read(c['cache_manifest'])
    longest = (0, None)
    for pair in pairs:
        if not pair['contrast']:
            continue
        row = by_id[pair['id']]
        prompt = tok.apply_chat_template(p.messages(row, 'B', True), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        before, after = prompt.split(p.MARKER)
        n = sum(len(tok(x, add_special_tokens=False).input_ids) for x in (before, after))
        n += math.ceil(cache[row['id']]['frames'] / 4)
        target = json.dumps({'text': pair['negative']}, ensure_ascii=False) + tok.eos_token
        n += len(tok(target, add_special_tokens=False).input_ids)
        assert n <= c['max_sequence'], row['id']
        longest = max(longest, (n, row['id']))
    assert longest[1], 'No pronunciation-contrast training pairs'
    smoke_ids = list(dict.fromkeys([c['smoke_ids'][0], c['sequence_audit']['train']['longest_id'],
                                   next(x['id'] for x in pairs if x['correct_input']), longest[1]]))
    stats = dict(train_n=len(pairs), correct_inputs=4254,
                 negative_eligible=sum(bool(x['negative']) for x in pairs),
                 contrast_selected=sum(x['contrast'] for x in pairs),
                 excluded_candidates={key: sum(x['rejected'][key] for x in pairs)
                                      for key in ('same_phonemes', 'non_han', 'too_far')},
                 longest_negative=longest)
    w.save(HERE / 'pair_audit.json', stats)
    sources = dict(c['sources'])
    for path in (Path(__file__), HERE / 'test_grounded.py', HERE / 'PLAN.md',
                 HERE / 'train_pairs.jsonl', OLD / 'protocol.json', OLD / 'summary.json'):
        sources[str(path.resolve())] = w.sha(path)
    import pypinyin
    c.pop('protocol_sha256', None)
    c.update(experiment='Pronunciation-grounded correction learning v1', version=1,
             sources=sources, parent_protocol=str(OLD / 'protocol.json'),
             train_pairs=str(HERE / 'train_pairs.jsonl'), pair_audit=stats,
             objective='CE(true)+0.5 CE(text)+0.1 KL(text.detach||true) on correct-input rows '
                       '+0.1 softplus(1-5*(gap_true-gap_zero.detach)) on fixed quarter eligible rows',
             text_weight=.5, consistency_weight=.1, contrast_weight=.1,
             contrast_beta=5., contrast_margin=1., smoke_ids=smoke_ids,
             trainable='Existing checkpoint30022 LoRA and Afull MLP; base9B/SenseVoice frozen',
             contrast_score='Sum answer-token log probabilities; both targets divided by their common max token count; includes JSON/EOS',
             negative_rule='Only actual visible train top1/first8 candidates, 1..4 normalized character edits, '
                           'Han-only strings with differing toneless sentence-context pinyin; no synthetic person-name negatives',
             auxiliary_rule='sha256(917|train_id) first8 hex modulo4==0, before any dev inference',
             pypinyin_version=pypinyin.__version__, dropout='Inherited LoRA dropout retained for ordinary CE; paired gradient calls disable Dropout only while retaining training/checkpointing; peers use eval/no-grad',
             selection='Fixed final2162 updates; full17294 examples exactly once; no dev selection',
             evaluation_modes=['text', 'true', 'zero', 'mismatch'],
             primary='Raw CER, hotword recall, correct-input corruption, S/D/I and paired95% bootstrap; no protection',
             success='True audio must improve over matched text/zero and old D/B baselines without losing recall; no guaranteed noninferiority',
             comparison='Same old30022+Afull initialization and data/order/update count as old B; extra forwards mean unequal FLOPs. Architecture/prompt unchanged.',
             limitations=[*c['limitations'],
                          'Train phonetic classes are dictionary proxies, not listened/aligned acoustic labels',
                          'Text/true consistency uses a stop-gradient peer, not an independent frozen teacher',
                          'Contrast excludes same-phoneme including tone-only alternatives; no general homophone acoustic distinction claimed',
                          'Three objectives tested together: cannot separately attribute their effects',
                          'Afull initializer retains prior pure-audio pretraining and checkpoint625 history'])
    (HERE / 'G').mkdir(exist_ok=False)
    w.save(HERE / 'protocol.json', c)
    w.save(HERE / 'status.json', dict(status='PREPARED', pair_audit=stats))
    print(json.dumps(stats), flush=True)


def configuration():
    old.configuration()  # Verify unchanged legacy sources, old data and every audio cache.
    c = w.read(HERE / 'protocol.json')
    for name, expected in c['sources'].items():
        assert w.sha(name) == expected, 'Locked source changed: ' + name
    c['protocol_sha256'] = w.sha(HERE / 'protocol.json')
    p = old.legacy(); p.ADAPTER = Path(c['adapter'])
    pairs = {x['id']: x for x in p.load(c['train_pairs'])}
    assert len(pairs) == c['train_n']
    return c, p, pairs


def target_values(logits, labels):
    """Causal target positions only; never score the prompt or acoustic prefix."""
    positions = labels[:, 1:].ne(-100)
    targets = labels[:, 1:][positions]
    selected = logits[:, :-1][positions].float()
    log_probs = selected.log_softmax(-1)
    nll = -log_probs.gather(-1, targets[:, None]).squeeze(-1)
    return nll.mean(), -nll.sum(), log_probs, len(targets)


def contrast_loss(gap_true, gap_zero, beta=5., margin=1.):
    import torch
    return torch.nn.functional.softplus(margin - beta * (gap_true - gap_zero.detach()))


def target_batch(rt, row, mode, negative=None):
    if negative is not None:
        row = dict(row, reference=negative)
    return rt.batch(row, training=True, mode=mode)


def target_score(rt, batch):
    # Avoid materializing vocabulary logits for the entire prompt. The extra
    # position is the last prefix token predicting the first answer token.
    n = int(batch['labels'].ne(-100).sum())
    args = {k: v for k, v in batch.items() if k != 'labels'}
    out = rt.model(**args, use_cache=False, logits_to_keep=n + 1)
    assert out.logits.shape[1] == n + 1
    return target_values(out.logits, batch['labels'][:, -(n + 1):])


def scoring_training_mode(rt, deterministic):
    import torch
    # model.eval() would disable gradient checkpointing and cause avoidable OOM.
    rt.model.train(); rt.connector.train()
    if deterministic:
        for module in rt.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()


def update(rt, rows, pairs, opt, sch):
    import torch
    c = rt.c
    opt.zero_grad(set_to_none=True)
    records = []
    for row in rows:
        pair = pairs[row['id']]
        rec = dict(id=row['id'], correct_input=pair['correct_input'], contrast=pair['contrast'])
        # Paired text CE prevents acoustic adaptation from replacing text learning.
        rt.model.train()
        text_batch = target_batch(rt, row, 'text')
        text_ce, _, _, _ = target_score(rt, text_batch)
        (c['text_weight'] * text_ce / len(rows)).backward()
        rec['text_ce'] = float(text_ce.detach())
        del text_ce, text_batch
        teacher = zero_gap = None
        if pair['correct_input'] or pair['contrast']:
            rt.model.eval(); rt.connector.eval()
            with torch.no_grad():
                if pair['correct_input']:
                    _, _, teacher, _ = target_score(rt, target_batch(rt, row, 'text'))
                    teacher = teacher.detach()
                if pair['contrast']:
                    _, pos, _, pos_n = target_score(rt, target_batch(rt, row, 'zero'))
                    _, neg, _, neg_n = target_score(rt, target_batch(rt, row, 'zero', pair['negative']))
                    scale = max(pos_n, neg_n)
                    zero_gap = ((pos - neg) / scale).detach()
                    rec['zero_gap'] = float(zero_gap)
        # CE keeps inherited dropout; paired contrast/KL use deterministic eval.
        scoring_training_mode(rt, pair['correct_input'] or pair['contrast'])
        true_ce, pos_score, true_log_probs, _ = target_score(rt, target_batch(rt, row, 'true'))
        loss = true_ce
        rec['true_ce'] = float(true_ce.detach())
        if teacher is not None:
            assert teacher.shape == true_log_probs.shape
            kl = torch.nn.functional.kl_div(true_log_probs, teacher, log_target=True, reduction='batchmean')
            assert torch.isfinite(kl) and kl >= -1e-5
            loss = loss + c['consistency_weight'] * kl
            rec['consistency_kl'] = float(kl.detach())
        if zero_gap is not None:
            _, neg_score, _, _ = target_score(rt, target_batch(rt, row, 'true', pair['negative']))
            gap = (pos_score - neg_score) / scale
            contrast = contrast_loss(gap, zero_gap, c['contrast_beta'], c['contrast_margin'])
            loss = loss + c['contrast_weight'] * contrast
            rec.update(true_gap=float(gap.detach()), contrast_loss=float(contrast.detach()))
        assert torch.isfinite(loss)
        (loss / len(rows)).backward()
        rec['objective'] = float(loss.detach()) + c['text_weight'] * rec['text_ce']
        records.append(rec)
        del loss, true_ce, true_log_probs, pos_score, teacher
        if zero_gap is not None:
            del neg_score, gap, contrast
        rt.model.train(); rt.connector.train()
    assert all(x.grad is None for x in rt.model.parameters() if not x.requires_grad)
    lg = sum(float(x.grad.detach().float().norm()) for _, x in rt.lora if x.grad is not None)
    cg = sum(float(x.grad.detach().float().norm()) for x in rt.connector.parameters() if x.grad is not None)
    assert math.isfinite(lg) and lg > 0 and math.isfinite(cg) and cg > 0
    norm = torch.nn.utils.clip_grad_norm_(rt.parameters, 1.)
    assert torch.isfinite(norm) and norm > 0
    opt.step(); sch.step()
    return dict(loss=sum(x['objective'] for x in records) / len(records), lora_grad=lg,
                connector_grad=cg, grad_norm=float(norm), sample_losses=records)


def smoke(reload_only=False):
    import torch
    from peft import get_peft_model_state_dict
    c, p, pairs = configuration(); d = HERE / 'G'; dest = d / 'smoke_checkpoint'
    rows = {r['id']: r for r in p.load(c['train'])}
    with w.gpu_guard(p):
        rt = old.Runtime(c, p, 'B', True, dest if reload_only else None)
        opt, sch = rt.optimizer()
        if reload_only:
            meta = old.restore(rt, opt, sch, dest)
            assert meta['step'] == len(c['smoke_ids']) and sch.last_epoch == meta['step']
            from safetensors.torch import load_file
            saved = load_file(str(dest / 'adapter/adapter_model.safetensors'))
            actual = get_peft_model_state_dict(rt.model)
            assert saved.keys() == actual.keys()
            assert all(torch.equal(v, actual[k].cpu()) for k, v in saved.items())
            cs = torch.load(dest / 'connector.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(v, rt.connector.state_dict()[k].cpu()) for k, v in cs.items())
            rt.model.eval(); rt.connector.eval()
            prediction = p.generate(rt.model, rt.tokenizer, rt.connector, rows[c['smoke_ids'][0]], 'B', 'true')
            assert prediction['model_output'] == w.read(d / 'smoke_before_reload.json')['model_output']
            rec = update(rt, [rows[c['smoke_ids'][0]]], pairs, opt, sch)
            w.save(d / 'smoke.json', dict(status='PASS', reload_exact=True, resumed_update=rec,
                                         protocol_sha256=c['protocol_sha256']))
        else:
            assert not dest.exists()
            before = {n: x.detach().cpu().clone() for n, x in rt.lora}
            cbefore = {n: x.detach().cpu().clone() for n, x in rt.connector.named_parameters()}
            frozen = {n: x._version for n, x in rt.model.named_parameters() if not x.requires_grad}
            records = []
            for uid in c['smoke_ids']:
                started = time.time()
                rec = update(rt, [rows[uid]], pairs, opt, sch)
                rec.update(seconds=time.time() - started, max_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
                records.append(rec)
                print(json.dumps(rec), flush=True)
            assert any(not torch.equal(before[n], x.cpu()) for n, x in rt.lora)
            assert any(not torch.equal(cbefore[n], x.cpu()) for n, x in rt.connector.named_parameters())
            assert all(x._version == frozen[n] for n, x in rt.model.named_parameters() if not x.requires_grad)
            old.checkpoint(rt, opt, sch, dest, len(records), len(records))
            rt.model.eval(); rt.connector.eval()
            pr = p.generate(rt.model, rt.tokenizer, rt.connector, rows[c['smoke_ids'][0]], 'B', 'true')
            w.save(d / 'smoke_before_reload.json', pr)
            w.save(d / 'smoke_training.json', dict(records=records, llm_trainable=sum(x.numel() for _, x in rt.lora)))
    print(json.dumps(dict(smoke='PASS', reload=reload_only)), flush=True)


def train():
    import torch
    c, p, pairs = configuration(); d = HERE / 'G'
    assert w.read(d / 'smoke.json')['status'] == 'PASS'
    assert w.read(d / 'smoke.json')['protocol_sha256'] == c['protocol_sha256']
    if (d / 'selected_checkpoint.json').exists():
        meta = old.check_checkpoint(Path(w.read(d / 'selected_checkpoint.json')['checkpoint']), c, 'B')
        assert (meta['step'], meta['examples']) == (2162, 17294)
        return
    previous = []
    for cp in (d / 'checkpoints').glob('step_*'):
        if (cp / 'metadata.json').exists():
            meta = old.check_checkpoint(cp, c, 'B'); previous.append((meta['step'], cp))
    start, cp = max(previous, default=(0, None))
    with w.gpu_guard(p):
        rt = old.Runtime(c, p, 'B', True, cp); opt, sch = rt.optimizer()
        if cp:
            old.restore(rt, opt, sch, cp)
        rows = p.load(c['train']); random.Random(c['seed']).shuffle(rows)
        (d / 'checkpoints').mkdir(exist_ok=True)
        began = time.time(); run_id = time.time_ns()
        w.save(d / f'training_started_{run_id}.json', dict(pid=os.getpid(), start_time=began, resume_step=start,
                                                        protocol_sha256=c['protocol_sha256']))
        with (d / f'training_{run_id}.jsonl').open('x', encoding='utf-8') as stream:
            for step in range(start, c['steps']):
                chunk = rows[step * c['accumulation']:(step + 1) * c['accumulation']]
                rec = update(rt, chunk, pairs, opt, sch)
                rec.update(status='TRAINING', step=step + 1, steps=c['steps'],
                           examples=min((step + 1) * c['accumulation'], len(rows)),
                           seconds=time.time() - began, resume_step=start,
                           max_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
                stream.write(json.dumps(rec) + '\n'); stream.flush()
                w.save(d / 'status.json', rec)
                if step == start or (step + 1) % 10 == 0:
                    print(json.dumps({k: v for k, v in rec.items() if k != 'sample_losses'}), flush=True)
                if (step + 1) % c['checkpoint_interval'] == 0 or step + 1 == c['steps']:
                    cp = d / 'checkpoints' / f'step_{step + 1}_{time.time_ns()}'
                    old.checkpoint(rt, opt, sch, cp, step + 1, rec['examples'])
                    w.save(d / 'latest_checkpoint.json', dict(checkpoint=str(cp), step=step + 1))
        w.save(d / 'selected_checkpoint.json', dict(checkpoint=str(cp), step=c['steps'], reason=c['selection']))
        w.save(d / 'status.json', dict(status='TRAINED_NOT_EVALUATED', checkpoint=str(cp)))


def evaluate():
    c, p, _ = configuration(); d = HERE / 'G'
    cp = Path(w.read(d / 'selected_checkpoint.json')['checkpoint'])
    meta = old.check_checkpoint(cp, c, 'B')
    assert (meta['step'], meta['examples']) == (2162, 17294)
    rows = p.load(c['dev']); dest = d / 'dev'; dest.mkdir(exist_ok=True)
    with w.gpu_guard(p):
        rt = old.Runtime(c, p, 'B', False, cp)
        for mode in c['evaluation_modes']:
            target = dest / f'{mode}.predictions.jsonl'
            if target.exists():
                rr = p.load(target); old.check_predictions(rr, rows, mode, c['audio_permutation'])
                assert len(rr) == len(rows)
                continue
            partial = Path(str(target) + '.partial')
            prior = p.load(partial) if partial.exists() else []
            old.check_predictions(prior, rows, mode, c['audio_permutation'])
            with partial.open('a', encoding='utf-8') as stream:
                for i in range(len(prior), len(rows)):
                    row = rows[i]
                    audio_id = c['audio_permutation'][row['id']] if mode == 'mismatch' else None
                    pr = p.generate(rt.model, rt.tokenizer, rt.connector, row, 'B', mode, audio_id)
                    stream.write(json.dumps(pr, ensure_ascii=False) + '\n'); stream.flush()
                    if (i + 1) % 25 == 0 or i + 1 == len(rows):
                        rec = dict(status='EVALUATING', mode=mode, done=i + 1, total=len(rows))
                        w.save(d / 'status.json', rec); print(json.dumps(rec), flush=True)
            rr = p.load(partial); old.check_predictions(rr, rows, mode, c['audio_permutation'])
            assert len(rr) == len(rows)
            partial.rename(target)
    metrics = {}
    for mode in c['evaluation_modes']:
        rr = p.load(dest / f'{mode}.predictions.jsonl')
        old.check_predictions(rr, rows, mode, c['audio_permutation'])
        m = old.score(p, rr); w.save(dest / f'{mode}.metrics.json', m); metrics['G_' + mode] = m
    for group, mode in [('D', 'text'), ('B', 'text'), ('B', 'true'), ('B', 'zero')]:
        rr = p.load(OLD / group / 'dev' / f'{mode}.predictions.jsonl')
        old.check_predictions(rr, rows, mode, c['audio_permutation'])
        metrics[group + '_' + mode] = old.score(p, rr)
    result = dict(summaries={k: v['summary'] for k, v in metrics.items()},
                  diagnostics={k: v['error_diagnostics'] for k, v in metrics.items()},
                  G_true_minus_control_95ci={k: p.bootstrap(metrics['G_true']['per_utterance'], v['per_utterance'])
                                           for k, v in metrics.items() if k != 'G_true'},
                  prediction_hashes={m: w.sha(dest / f'{m}.predictions.jsonl') for m in c['evaluation_modes']},
                  protocol_sha256=c['protocol_sha256'], no_output_repairs=True,
                  limitations=c['limitations'])
    w.save(HERE / 'summary.json', result)
    w.save(d / 'status.json', dict(status='COMPLETE'))
    w.save(HERE / 'status.json', dict(status='ALL_DONE', completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def pipeline():
    import fcntl
    with (HERE / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in ('smoke', 'reload-smoke', 'train', 'eval'):
            if task in ('smoke', 'reload-smoke') and (HERE / 'G/smoke.json').exists():
                continue
            if task == 'smoke' and (HERE / 'G/smoke_checkpoint/metadata.json').exists():
                continue
            w.save(HERE / 'status.json', dict(status='RUNNING', phase=task, pid=os.getpid(), updated=time.time()))
            with (HERE / f'{task}_{time.time_ns()}.log').open('x') as stream:
                subprocess.run([sys.executable, '-u', str(Path(__file__)), task],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert w.read(HERE / 'status.json')['status'] == 'ALL_DONE'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'smoke', 'reload-smoke', 'train', 'eval', 'pipeline'])
    args = ap.parse_args()
    try:
        if args.command == 'prepare': prepare()
        elif args.command == 'smoke': smoke()
        elif args.command == 'reload-smoke': smoke(True)
        elif args.command == 'train': train()
        elif args.command == 'eval': evaluate()
        else: pipeline()
    except BaseException:
        err = dict(status='FAILED', command=args.command, error=traceback.format_exc(), time=time.time())
        w.save(HERE / f'failure_{time.time_ns()}.json', err); w.save(HERE / 'status.json', err)
        raise
