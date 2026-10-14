#!/usr/bin/env python3
"""Diagnostic staged acoustic coupling pilot. No test-set tuning or fallback edits."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path('/root/autodl-tmp')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
BASE = ROOT / 'cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B'
ADAPTER = ROOT / 'src/logs/hotword_lora_9b_20260908/scheduled/high_lr3e-5/adapter/checkpoint-625'
SOURCE = ROOT / 'src/logs/cb_sensevoice_w14_train_evidence_full.jsonl.gz'
DEV_SOURCE = ADAPTER.parents[1] / 'full_625.predictions.jsonl'
MANIFESTS = ROOT / '.dsh_checks/rerank'
ALIGNED = ROOT / 'datasets/aishell/data_aishell_sensevoice/hotword/dev/aligned.txt'
SENSE = Path('/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master')
MARKER = '@@CONTINUOUS_AUDIO@@'
SEED = 917


def load(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert len(load(tmp)) == len(rows)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluator():
    spec = importlib.util.spec_from_file_location('restore_eval', ROOT / '.dsh_checks/restore_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank(uid):
    return hashlib.sha256(f'{SEED}|{uid}'.encode()).hexdigest()


def unique(rows, field):
    out = {str(r[field]): r for r in rows}
    assert len(out) == len(rows), f'duplicate {field}'
    return out


def prepare():
    assert not (RUN / 'protocol.json').exists(), 'immutable run already prepared'
    ev = evaluator()
    trmap = unique(load(MANIFESTS / 'aishell_train_manifest.jsonl'), 'key')
    dvmap = unique(load(MANIFESTS / 'aishell_dev_manifest.jsonl'), 'id')
    dev = []
    for row in load(DEV_SOURCE):
        uid = str(row['id'])
        m = dvmap[uid]
        assert ev.norm(m['reference']) == ev.norm(row['reference'])
        dev.append(dict(id=uid, wav=m['wav'], reference=row['reference'], input=row['input'], split='dev'))
    assert len(dev) == 1334
    unique(dev, 'id')
    devrefs = {ev.norm(r['reference']) for r in dev}
    devuids = {r['id'] for r in dev}
    candidates, excluded = [], 0
    source = load(SOURCE)
    for row in source:
        m = trmap[f"{row['split']}|{row['id']}"]
        assert ev.norm(m['reference']) == ev.norm(row['reference'])
        assert m['id'] not in devuids
        if ev.norm(row['reference']) in devrefs:
            excluded += 1
            continue
        candidates.append(dict(id=m['id'], wav=m['wav'], reference=row['reference'],
                               input=row['input'], split='train'))
    unique(candidates, 'id')
    train = sorted(candidates, key=lambda r: rank(r['id']))[:4096]
    assert len(train) == 4096
    screen = sorted(dev, key=lambda r: rank(r['id']))[:64]
    for row in train + dev:
        assert Path(row['wav']).is_file() and Path(row['wav']).stem == row['id']
        assert ev.norm(row['reference']) and ev.norm(row['input']['asr_top1'])
    RUN.mkdir(parents=True, exist_ok=True)
    for name, rows in [('train', train), ('dev', dev), ('screen', screen), ('smoke', train[:2] + screen[:2])]:
        save_rows(RUN / f'{name}.jsonl', rows)
    paths = [SOURCE, DEV_SOURCE, MANIFESTS / 'aishell_train_manifest.jsonl',
             MANIFESTS / 'aishell_dev_manifest.jsonl', ALIGNED, Path(__file__),
             ROOT / '.dsh_checks/restore_eval.py', ADAPTER / 'adapter_config.json',
             ADAPTER / 'adapter_model.safetensors']
    protocol = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), label='AISHELL diagnostic pilot, not held-out test',
        seed=SEED, base=str(BASE), adapter=str(ADAPTER), train_n=4096, screen_n=64, dev_n=1334,
        steps_A=512, steps_B=512, microbatch=1, accumulation=8, lr_A=2e-4, lr_B=1e-5,
        lr_connector_B=5e-5, warmup_steps=32, max_sequence=4096, max_new_tokens=96,
        decode='greedy; enable_thinking=False; batch1; no output repair',
        connector='mean-pool consecutive 4 frames incl last partial group; LN512 Linear512-1024 GELU Linear1024-hidden; fp32 params',
        acoustic='SenseVoiceSmall encoder continuous valid frames excluding 4 queries; float16; zh/woitn',
        stage_A='audio-only transcription; frozen original 9B + existing adapter; connector only',
        stage_B='audio + whitelisted CBSense text; connector + continue existing LoRA; frontend frozen',
        gate_A='all 64 true-audio outputs parsed and nonempty; true CER strictly below zero and deterministic mismatch CER',
        gate_B='only final checkpoint; raw true-audio CER lower AND hits >= matched protected-text baseline; also lower CER than zero/mismatch',
        failure='stop on exception/nonfinite loss/missing ID; no omitted rows or automatic parameter changes',
        source_train_n=len(source), excluded_duplicate_dev_references=excluded,
        source_error_rate=sum(ev.norm(r['reference']) != ev.norm(r['input']['asr_top1']) for r in source)/len(source),
        selected_error_rate=sum(ev.norm(r['reference']) != ev.norm(r['input']['asr_top1']) for r in train)/len(train),
        context_caveat='Inherited contextual evidence protocol; oracle_hotwords/keyword_mentions/references not exposed to prompt. External vocabulary provenance not re-established; diagnostic only.',
        normalization='shared restore_eval: NFKC, remove spaces/punctuation, NO traditional-simplified conversion',
        sources={str(p): digest(p) for p in paths},
        data_hashes={n: digest(RUN / f'{n}.jsonl') for n in ['train','dev','screen','smoke']})
    save(RUN / 'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def export_audio(split):
    import numpy as np
    import torch
    from funasr import AutoModel
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
    torch.set_num_threads(6)
    wrapper = AutoModel(model=str(SENSE), device='cuda:0', disable_update=True)
    model, frontend = wrapper.model.eval(), wrapper.kwargs['frontend']
    cache = RUN / 'audio'
    cache.mkdir(exist_ok=True)
    rows = load(RUN / f'{split}.jsonl')
    started = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            path = cache / f"{row['id']}.npy"
            if path.exists():
                x = np.load(path)
                assert x.ndim == 2 and x.shape[1] == 512 and len(x) > 0 and np.isfinite(x).all()
                continue
            audio = load_audio_text_image_video(row['wav'], fs=frontend.fs, audio_fs=None, data_type='sound', tokenizer=None)
            speech, lengths = extract_fbank(audio, data_type='sound', frontend=frontend)
            speech, lengths = speech.cuda(), lengths.cuda()
            valid = int(lengths[0])
            def query(ids):
                return model.embed(torch.tensor([ids], device='cuda', dtype=torch.long))
            x = torch.cat([query([model.lid_dict['zh']]), query([1,2]),
                           query([model.textnorm_dict['woitn']]), speech], dim=1)
            enc, out_lengths = model.encoder(x, lengths + 4)
            if isinstance(enc, tuple):
                enc = enc[0]
            # SenseVoice encoder is length preserving; never trim by reference/CTC alignment.
            assert int(out_lengths[0]) == valid + 4 and enc.shape[1] >= valid + 4
            data = enc[0,4:valid+4].float().cpu().numpy().astype(np.float16)
            assert data.shape == (valid,512) and np.isfinite(data).all()
            with open(str(path) + '.tmp', 'wb') as f:
                np.save(f, data)
            os.replace(str(path)+'.tmp', path)
            if i % 50 == 0:
                print(json.dumps(dict(export=split, done=i+1, total=len(rows), seconds=time.time()-started)), flush=True)
    save(RUN / f'export_{split}.json', dict(rows=len(rows), seconds=time.time()-started,
         cache=[dict(id=r['id'], shape=list(np.load(cache/f"{r['id']}.npy", mmap_mode='r').shape),
                     sha256=digest(cache/f"{r['id']}.npy")) for r in rows]))


def evidence(row):
    ev = evaluator()
    inp = row['input']
    # Explicit whitelist: do not pass reference-aware metadata or prior predictions.
    return {'asr_top1': inp['asr_top1'], 'nbest': ev.texts_of(inp.get('nbest'))[:8],
            'hotwords': (ev.texts_of(inp.get('prompt_hotwords')) or ev.texts_of(inp.get('hotwords')))[:32]}


def messages(row, stage, acoustic):
    system = '你是中文语音转写助手。仅输出JSON对象，格式为{"text":"最终转写"}，不输出解释。'
    if stage == 'A':
        user = '请准确转写下面的语音，不添加声音中没有的内容。\n语音：' + MARKER
    else:
        user = '请校正语音识别结果。根据实际说话内容选择用词，不要仅为语言流畅改写。热词只是候选提示，不必强行加入。\n'
        if acoustic:
            user += '语音：' + MARKER + '\n'
        user += '识别证据：' + json.dumps(evidence(row), ensure_ascii=False)
    return [{'role':'system','content':system}, {'role':'user','content':user}]


def load_model(stage, checkpoint=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(SEED)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, device_map={'':'cuda:0'})
    adapter = Path(checkpoint)/'adapter' if checkpoint and (Path(checkpoint)/'adapter').exists() else ADAPTER
    model = PeftModel.from_pretrained(model, adapter, is_trainable=training and stage == 'B')
    if not (training and stage == 'B'):
        for p in model.parameters():
            p.requires_grad_(False)
    connector = torch.nn.Sequential(torch.nn.LayerNorm(512), torch.nn.Linear(512,1024),
                                   torch.nn.GELU(), torch.nn.Linear(1024,int(model.config.hidden_size))).cuda()
    if checkpoint:
        connector.load_state_dict(torch.load(Path(checkpoint)/'connector.pt', map_location='cuda', weights_only=True))
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache = False
        # Checkpointing requires model.training. Disable all frozen backbone dropout in A.
        model.train()
        if stage == 'A':
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
        connector.train()
    else:
        model.eval()
        connector.eval()
        model.config.use_cache = True
    print(json.dumps(dict(stage=stage, training=training, llm_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        connector_params=sum(p.numel() for p in connector.parameters()))), flush=True)
    return model, tokenizer, connector


def acoustic_tensor(uid, connector, mode):
    import numpy as np
    import torch
    x = torch.from_numpy(np.load(RUN/'audio'/f'{uid}.npy').astype(np.float32)).cuda()
    # Pool every frame exactly once, including a short final group.
    n = len(x)
    full = n // 4
    parts = []
    if full:
        parts.append(x[:full*4].reshape(full,4,512).mean(1))
    if n % 4:
        parts.append(x[full*4:].mean(0,keepdim=True))
    y = connector(torch.cat(parts,dim=0))
    return torch.zeros_like(y) if mode == 'zero' else y


def batch(model, tokenizer, connector, row, stage, mode='true', audio_id=None, training=False):
    import torch
    acoustic = mode != 'text'
    prompt = tokenizer.apply_chat_template(messages(row,stage,acoustic), tokenize=False,
                 add_generation_prompt=True, enable_thinking=False)
    embed = model.get_input_embeddings()
    def text_embed(s):
        ids = tokenizer(s,add_special_tokens=False,return_tensors='pt').input_ids.cuda()
        return embed(ids)[0]
    if acoustic:
        assert prompt.count(MARKER) == 1
        before, after = prompt.split(MARKER)
        ac = acoustic_tensor(audio_id or row['id'], connector, mode).to(embed.weight.dtype)
        pieces = [text_embed(before),ac,text_embed(after)]
    else:
        pieces = [text_embed(prompt)]
    prefix = sum(len(x) for x in pieces)
    labels = None
    if training:
        target = json.dumps({'text':row['reference']},ensure_ascii=False) + tokenizer.eos_token
        ids = tokenizer(target,add_special_tokens=False,return_tensors='pt').input_ids.cuda()[0]
        pieces.append(embed(ids))
        labels = torch.cat([torch.full((prefix,),-100,device='cuda',dtype=torch.long),ids]).unsqueeze(0)
        assert (labels[0,:prefix] == -100).all() and (labels[0,prefix:] != -100).all()
    inputs = torch.cat(pieces,dim=0).unsqueeze(0)
    assert inputs.shape[1] <= 4096, f"sequence too long: {row['id']} {inputs.shape[1]}"
    out = dict(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2],device='cuda',dtype=torch.long))
    if labels is not None:
        out['labels'] = labels
    return out


def parse_prediction(text):
    try:
        value = json.loads(text.strip())
        if isinstance(value,dict) and isinstance(value.get('text'),str):
            return value['text'], True
    except (ValueError,TypeError):
        pass
    # Malformed responses are scored as empty, never silently copied from top1.
    return '', False


def generate(model, tokenizer, connector, row, stage, mode, audio_id=None, max_tokens=96):
    import torch
    with torch.no_grad():
        b = batch(model,tokenizer,connector,row,stage,mode,audio_id)
        ids = model.generate(**b, do_sample=False, max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    # For inputs_embeds only, HF returns generated tokens, not a prompt-ID prefix.
    assert ids.shape[1] <= max_tokens, 'unexpected inputs_embeds generation API'
    raw = tokenizer.decode(ids[0],skip_special_tokens=True)
    pred, parsed = parse_prediction(raw)
    return dict(id=row['id'],reference=row['reference'],input=row['input'],prediction=pred,
                model_output=raw, parsed=parsed, audio_id=audio_id or row['id'], condition=mode)


def save_checkpoint(model, connector, stage, step):
    import torch
    dest = RUN / f'{stage}_{step}'
    dest.mkdir(exist_ok=False)
    torch.save(connector.state_dict(), dest/'connector.pt')
    if stage == 'B':
        model.save_pretrained(dest/'adapter')
    save(dest/'metadata.json',dict(stage=stage,step=step,base=str(BASE),initial_adapter=str(ADAPTER),complete=True))
    return dest


def train(stage):
    import torch
    rows = load(RUN/'train.jsonl')
    random.Random(SEED).shuffle(rows)
    checkpoint = RUN/'A_512' if stage == 'B' else None
    model, tokenizer, connector = load_model(stage,checkpoint,True)
    groups = [dict(params=list(connector.parameters()),lr=2e-4 if stage=='A' else 5e-5)]
    llm_params = [p for p in model.parameters() if p.requires_grad]
    if llm_params:
        groups.append(dict(params=llm_params,lr=1e-5))
    assert bool(llm_params) == (stage == 'B')
    optimizer = torch.optim.AdamW(groups,weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s: min((s+1)/32,1)*max(0.0,(512-s)/480) if s>=32 else (s+1)/32)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i,row in enumerate(rows):
        b = batch(model,tokenizer,connector,row,stage,training=True)
        loss = model(**b,use_cache=False).loss
        assert torch.isfinite(loss), f'nonfinite loss {row["id"]}'
        (loss/8).backward()
        losses.append(float(loss.detach()))
        if (i+1)%8 == 0:
            step=(i+1)//8
            grad = torch.nn.utils.clip_grad_norm_(list(connector.parameters())+llm_params,1.0)
            assert torch.isfinite(grad) and float(grad)>0
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step == 1 or step%10 == 0:
                print(json.dumps(dict(stage=stage,step=step,steps=512,loss=sum(losses)/len(losses),
                      grad_norm=float(grad),seconds=time.time()-started,
                      gpu_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
                losses=[]
            if step in (256,512):
                save_checkpoint(model,connector,stage,step)
    save(RUN/f'train_{stage}.json',dict(steps=512,seconds=time.time()-started,final=str(RUN/f'{stage}_512')))


def smoke():
    import torch
    rows=load(RUN/'smoke.jsonl')
    model,tokenizer,connector=load_model('A',training=True)
    assert not any(p.requires_grad for p in model.parameters())
    started=time.time()
    loss=model(**batch(model,tokenizer,connector,rows[0],'A',training=True),use_cache=False).loss
    loss.backward()
    grad=sum(float(p.grad.norm()) for p in connector.parameters() if p.grad is not None)
    assert grad>0 and math.isfinite(grad)
    assert all(p.grad is None for p in model.parameters())
    # Exercise the stage-B prompt path too; full LoRA training is tested in a separate process.
    b=batch(model,tokenizer,connector,rows[1],'B',training=True)
    assert b['labels'].shape == b['attention_mask'].shape
    connector.zero_grad(set_to_none=True)
    model.eval(); connector.eval()
    model.gradient_checkpointing_disable()
    pred=generate(model,tokenizer,connector,rows[2],'A','true',max_tokens=16)
    checkpoint=RUN/'smoke_connector.pt'
    torch.save(connector.state_dict(),checkpoint)
    expected={k:v.detach().clone() for k,v in connector.state_dict().items()}
    connector.load_state_dict(torch.load(checkpoint,weights_only=True))
    assert all(torch.equal(v,connector.state_dict()[k]) for k,v in expected.items())
    result=dict(status='PASS',code_sha256=digest(__file__),loss=float(loss.detach()),connector_gradient_norm=grad,
                frozen_backbone_gradient=True,generation=pred['model_output'],
                seconds=time.time()-started,gpu_gb=torch.cuda.max_memory_allocated()/1e9)
    save(RUN/'smoke.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def smoke_b():
    import torch
    model,tokenizer,connector=load_model('B',training=True)
    row=load(RUN/'smoke.jsonl')[0]
    loss=model(**batch(model,tokenizer,connector,row,'B',training=True),use_cache=False).loss
    loss.backward()
    llm=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert llm and all('lora_' in n for n,p in llm)
    grad=sum(float(p.grad.norm()) for _,p in llm if p.grad is not None)
    assert math.isfinite(grad) and grad>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer=torch.optim.AdamW([p for _,p in llm]+list(connector.parameters()),lr=1e-5)
    optimizer.step()
    save(RUN/'smoke_B.json',dict(status='PASS',code_sha256=digest(__file__),loss=float(loss.detach()),lora_gradient_norm=grad,
                               gpu_gb=torch.cuda.max_memory_allocated()/1e9))
    print('STAGE B BACKWARD PASS',flush=True)


def permutation(rows):
    import numpy as np
    ev=evaluator()
    ordered=sorted(rows,key=lambda r:(np.load(RUN/'audio'/f"{r['id']}.npy",mmap_mode='r').shape[0],r['id']))
    # Circular adjacent shift in duration order, forbid self and same-reference pairs.
    for shift in range(1,len(rows)):
        other=ordered[shift:]+ordered[:shift]
        if all(a['id']!=b['id'] and ev.norm(a['reference'])!=ev.norm(b['reference']) for a,b in zip(ordered,other)):
            return {a['id']:b['id'] for a,b in zip(ordered,other)}
    raise RuntimeError('No legal audio permutation')


def metrics(rows, protected=False):
    ev=evaluator()
    annotations={}
    for line in ALIGNED.read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('\t')
        if len(fields)>=2:
            annotations.setdefault(fields[1].strip(),[]).append(ev.norm(fields[0]))
    totals=dict(n=len(rows),chars=0,errors=0,input_errors=0,hits=0,mentions=0,lost=0,gained=0,
                improved=0,worsened=0,unchanged=0,exact=0,empty=0,parse_failures=0)
    per=[]
    unique(rows,'id')
    for r in rows:
        inp=r['input']; ref=ev.norm(r['reference']); p=ev.norm(r['prediction']); x=ev.norm(inp['asr_top1'])
        assert ref
        if protected:
            terms=ev.texts_of(inp.get('prompt_hotwords')) or ev.texts_of(inp.get('hotwords'))
            pool=ev.texts_of(inp.get('nbest'))+ev.texts_of((inp.get('cbwhisper') or {}).get('candidates'))
            zones=ev.spans_in(x,[t for t in terms if t in x and any(t in c for c in pool)])
            p=ev.restore(x,p,zones)
        err=ev.edit_distance(list(ref),list(p)); ie=ev.edit_distance(list(ref),list(x))
        terms=annotations.get(r['id'],[])
        hits=sum(t in p for t in terms)
        for key,val in dict(chars=len(ref),errors=err,input_errors=ie,hits=hits,mentions=len(terms),
            lost=sum(t in x and t not in p for t in terms),gained=sum(t not in x and t in p for t in terms),
            improved=int(err<ie),worsened=int(err>ie),unchanged=int(err==ie),exact=int(err==0),
            empty=int(not p),parse_failures=int(not r.get('parsed',True))).items():
            totals[key]+=val
        per.append(dict(id=r['id'],chars=len(ref),errors=err,hits=hits,mentions=len(terms)))
    totals['cer']=totals['errors']/totals['chars']
    totals['recall']=totals['hits']/totals['mentions'] if totals['mentions'] else None
    totals['input_cer']=totals['input_errors']/totals['chars']
    totals['cer_delta_pp']=100*(totals['cer']-totals['input_cer'])
    totals['relative_cer_change']=(totals['errors']-totals['input_errors'])/totals['input_errors'] if totals['input_errors'] else None
    return dict(summary=totals,per_utterance=per)


def evaluate(stage,split):
    rows=load(RUN/f'{split}.jsonl')
    checkpoint=None if stage=='text' else RUN/f'{stage}_512'
    model,tokenizer,connector=load_model('B' if stage=='text' else stage,checkpoint)
    modes=['text'] if stage=='text' else ['true','zero','mismatch']
    mapping=permutation(rows)
    save(RUN/f'{stage}_{split}_permutation.json',mapping)
    started=time.time()
    for mode in modes:
        predictions=[]
        path=RUN/f'{stage}_{split}_{mode}.predictions.jsonl'
        assert not path.exists(),f'will not overwrite {path}'
        # Keep progress evidence on failure but never mark partial output complete.
        with open(str(path)+'.partial','w',encoding='utf-8') as f:
            for i,row in enumerate(rows):
                pred=generate(model,tokenizer,connector,row,'B' if stage=='text' else stage,mode,
                              mapping[row['id']] if mode=='mismatch' else None)
                predictions.append(pred)
                f.write(json.dumps(pred,ensure_ascii=False)+'\n'); f.flush()
                if i%16==0:
                    print(json.dumps(dict(stage=stage,split=split,mode=mode,done=i+1,total=len(rows),seconds=time.time()-started)),flush=True)
        assert [r['id'] for r in predictions]==[r['id'] for r in rows]
        assert len(load(str(path)+'.partial'))==len(rows)
        os.replace(str(path)+'.partial',path)
        result=metrics(predictions)
        result['seconds_since_start']=time.time()-started
        save(RUN/f'{stage}_{split}_{mode}.metrics.json',result)
        print(json.dumps(result['summary']),flush=True)
        if stage!='A':
            save(RUN/f'{stage}_{split}_{mode}.protected.metrics.json',metrics(predictions,True))


def bootstrap(a,b):
    import numpy as np
    assert [x['id'] for x in a]==[x['id'] for x in b]
    rng=np.random.default_rng(SEED)
    n=len(a); delta=[]; recall=[]
    da=np.array([x['errors']-y['errors'] for x,y in zip(a,b)])
    dh=np.array([x['hits']-y['hits'] for x,y in zip(a,b)])
    chars=np.array([x['chars'] for x in a]); mentions=np.array([x['mentions'] for x in a])
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        delta.append(100*da[idx].sum()/chars[idx].sum())
        if mentions[idx].sum():
            recall.append(100*dh[idx].sum()/mentions[idx].sum())
    return dict(cer_delta_pp_95ci=np.quantile(delta,[.025,.975]).tolist(),
                recall_delta_pp_95ci=np.quantile(recall,[.025,.975]).tolist() if recall else None,
                warning='point noninferiority is not a statistical noninferiority guarantee')


def get_metric(stage,split,mode,protected=False):
    suffix='.protected' if protected else ''
    return json.loads((RUN/f'{stage}_{split}_{mode}{suffix}.metrics.json').read_text())


def gate(stage,split):
    true=get_metric(stage,split,'true')['summary']
    zero=get_metric(stage,split,'zero')['summary']
    mismatch=get_metric(stage,split,'mismatch')['summary']
    passed=true['parse_failures']==0 and true['empty']==0 and true['cer']<min(zero['cer'],mismatch['cer'])
    result=dict(stage=stage,split=split,true=true,zero=zero,mismatch=mismatch,passed=passed)
    if stage=='B':
        baseline=get_metric('text',split,'text',True)
        legacy=get_metric('legacy',split,'text',True)
        result['protected_text']=baseline['summary']
        result['legacy_protected_text']=legacy['summary']
        result['passed']=passed and all(true['cer']<b['summary']['cer'] and true['hits']>=b['summary']['hits'] for b in [baseline,legacy])
        result['paired_bootstrap']=bootstrap(get_metric(stage,split,'true')['per_utterance'],baseline['per_utterance'])
    save(RUN/f'gate_{stage}_{split}.json',result)
    return result


def pipeline():
    import fcntl
    lock=open(RUN/'pipeline.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    def command(*args):
        name='_'.join(args)
        save(RUN/'status.json',dict(status='RUNNING',phase=name,pid=os.getpid(),started=started,updated=time.time()))
        with open(RUN/f'{name}.log','x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        protocol=json.loads((RUN/'protocol.json').read_text())
        implementation=json.loads((RUN/'implementation_lock.json').read_text())
        assert digest(__file__)==implementation['code_sha256'], 'code changed after implementation lock'
        for path,sha in implementation['frontend_hashes'].items():
            assert digest(path)==sha, 'frontend changed'
        for name,sha in protocol['data_hashes'].items():
            assert digest(RUN/f'{name}.jsonl')==sha
        assert json.loads((RUN/'smoke.json').read_text())['status']=='PASS'
        assert json.loads((RUN/'smoke_B.json').read_text())['status']=='PASS'
        command('export','train'); command('export','dev')
        command('train','A'); command('eval','A','screen')
        if not gate('A','screen')['passed']:
            save(RUN/'status.json',dict(status='COMPLETED_NEGATIVE_STAGE_A',seconds=time.time()-started,
                 reason='Audio utilization gate failed; no automatic stage B or expanded training'))
            return
        command('eval','text','screen')
        command('train','B'); command('eval','B','screen')
        candidate=gate('B','screen')
        if not candidate['passed']:
            save(RUN/'status.json',dict(status='COMPLETED_NEGATIVE_STAGE_B_SCREEN',seconds=time.time()-started,
                 reason='No checkpoint met frozen CER/recall/audio-use constraints'))
            return
        save(RUN/'selected_checkpoint.json',dict(checkpoint=str(RUN/'B_512'),reason='fixed final checkpoint passed screen gate',
              label='full-dev confirmation only, no held-out test'))
        command('eval','text','dev'); command('eval','B','dev')
        result=gate('B','dev')
        save(RUN/'status.json',dict(status='COMPLETED',passed=result['passed'],seconds=time.time()-started))
    except BaseException:
        save(RUN/'status.json',dict(status='FAILED',error=traceback.format_exc(),seconds=time.time()-started))
        raise


def lock_implementation():
    """Freeze execution code after engineering smoke checks, before any pilot training."""
    import importlib.metadata
    import shutil
    assert not (RUN/'status.json').exists() and not list(RUN.glob('A_*'))
    assert not (RUN/'implementation_lock.json').exists()
    for name in ['smoke','smoke_B']:
        s=json.loads((RUN/f'{name}.json').read_text())
        assert s['status']=='PASS' and s['code_sha256']==digest(__file__)
    legacy=unique(load(DEV_SOURCE),'id')
    for split in ['screen','dev']:
        rows=[legacy[r['id']] for r in load(RUN/f'{split}.jsonl')]
        for protected in [False,True]:
            suffix='.protected' if protected else ''
            save(RUN/f'legacy_{split}_text{suffix}.metrics.json',metrics(rows,protected))
    shutil.copy2(__file__,RUN/'implementation.py.snapshot')
    save(RUN/'implementation_lock.json',dict(code_sha256=digest(__file__),locked=time.time(),
         changes_since_initial_prepare=['Pin cached frontend path','Test LoRA optimizer step',
            'Require improvement over both matched prompt baseline and original protected baseline'],
         baseline_note='Legacy original-prompt predictions retained as a second strong system comparator',
         frontend_hashes={str(SENSE/f):digest(SENSE/f) for f in ['model.pt','config.yaml','am.mvn']},
         runtime={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','funasr','numpy']}))
    print('IMPLEMENTATION LOCKED BEFORE PILOT TRAINING',flush=True)


def selftest():
    ev=evaluator()
    assert parse_prediction('{"text":"你好"}')==('你好',True)
    assert parse_prediction('你好')==('',False)
    assert ev.norm('Ａ，你 好！')=='A你好'
    assert ev.edit_distance(list('你好'),list('你们'))==1
    assert ev.restore('乔加伟抵达北京','乔佳伟抵达北京',[(0,3)])=='乔加伟抵达北京'
    row=dict(reference='SECRET_REFERENCE',input={'asr_top1':'语音识别','nbest':['语音识别'],
            'oracle_hotwords':['SECRET_ORACLE'],'keyword_mentions':['SECRET_MENTIONS']})
    s=json.dumps(messages(row,'B',True),ensure_ascii=False)
    assert 'SECRET_' not in s
    assert json.dumps(messages(row,'A',True),ensure_ascii=False).count(MARKER)==1
    assert 'asr_top1' not in json.dumps(messages(row,'A',True))
    for name in ['train','dev','screen','smoke']:
        unique(load(RUN/f'{name}.jsonl'),'id')
    assert set(r['id'] for r in load(RUN/'train.jsonl')).isdisjoint(r['id'] for r in load(RUN/'dev.jsonl'))
    print('CPU SELFTEST PASS',flush=True)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare': prepare()
    elif cmd=='export': export_audio(sys.argv[2])
    elif cmd=='smoke': smoke()
    elif cmd=='smoke-b': smoke_b()
    elif cmd=='train': train(sys.argv[2])
    elif cmd=='eval': evaluate(sys.argv[2],sys.argv[3])
    elif cmd=='pipeline': pipeline()
    elif cmd=='lock': lock_implementation()
    elif cmd=='selftest': selftest()
    else: raise ValueError(cmd)
