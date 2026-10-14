"""Frozen old-best9B local candidate review with sealed native ASR evidence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
EVID=Path('/root/autodl-tmp/src/experiments/acoustic_review_20261005')
OLD=HERE.parent/'joint_lora_20260923'
BASE=Path('/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B')
sys.path.insert(0,str(EVID))
import review as evidence
sys.path.insert(0,str(HERE.parent/'text_residual_coupling_20260920'))
import workflow as w
from diagnostics import diagnose

SYSTEM='你是中文语音转写复核助手。仅输出JSON对象，格式为{"text":"选中的候选片段或证据不足"}，不输出解释。'
INSTRUCTION=('只复核本次指定的一个修改片段，不改写其他位置。候选A和B是两个可能的转写，顺序随机，不代表优先级。'
 '请结合识别输入、初稿、上下文、N-best及热词，判断该片段更应采用哪一个候选。'
 '发音由字典给出，可能有多音字问题；热词只是候选，不应强行加入。'
 '声学字段是SenseVoice对两个完整候选句的CTC对数似然，两句只有指定片段不同；数值越大表示支持较强。'
 '字符分数包含拼写偏好，发音合并分数也不是校准置信度；识别器可能维持自己原来的错误，不能仅按分数大小选择。'
 '语言自然也不能证明声音说了这个词。声学证据缺失时不要编造；无法可靠判断时输出“证据不足”，维持初稿。'
 'text只写候选A的片段、候选B的片段，或“证据不足”；也允许写对应的完整候选句。')


def seedbit(edit_id):
    return int(hashlib.sha256(('917|candidate_order|'+edit_id).encode()).hexdigest()[:16],16)%2


def whitelist(inp):
    def texts(value):
        out=[]
        for x in value or []:
            if isinstance(x,str):out.append(x)
            elif isinstance(x,dict) and isinstance(x.get('text'),str):out.append(x['text'])
        return out
    return dict(asr_top1=inp['asr_top1'],nbest=texts(inp.get('nbest'))[:8],
                hotwords=(texts(inp.get('prompt_hotwords')) or texts(inp.get('hotwords')))[:32])


def make_query(q,e,inp,scores):
    original=dict(role='original',text=e['original'],sentence=e['candidate_original_context'],pinyin=e['original_phones'])
    modified=dict(role='modified',text=e['modified'],sentence=e['candidate_modified_context'],pinyin=e['modified_phones'])
    a,b=(modified,original) if seedbit(e['edit_id']) else (original,modified)
    return dict(id=q['id'],edit_id=e['edit_id'],output_span=e['output_span'],draft=q['normalized_output'],
                input_evidence=whitelist(inp),A=a,B=b,evidence=scores)


def messages(q,mode):
    item=dict(识别证据=q['input_evidence'],初稿=q['draft'],复核片段位置=q['output_span'],候选={})
    for key in ('A','B'):
        cand=q[key];acoustic={}
        for method,label in [('native','字符CTC_logp'),('phonetic_marginal','发音合并CTC_logp')]:
            value=None if mode=='none' else q['evidence'][mode][method][cand['role']]['score']
            acoustic[label]='未提供' if mode=='none' else (round(value,4) if value is not None else '不可评分')
        item['候选'][key]=dict(片段=cand['text'],拼音=cand['pinyin'],完整候选句=cand['sentence'],声学证据=acoustic)
    return [dict(role='system',content=SYSTEM),dict(role='user',content=INSTRUCTION+'\n'+json.dumps(item,ensure_ascii=False))]


def parse(raw,q):
    try:
        value=json.loads(raw.strip())
        if not isinstance(value,dict) or not isinstance(value.get('text'),str):return dict(valid=False,reason='non_text_JSON')
        text=value['text'].strip()
        if text in ('证据不足','U'):return dict(valid=True,choice='U',selected_role='modified',abstained=True)
        for key in ('A','B'):
            cand=q[key]
            if text==key or evidence.norm(text) in (evidence.norm(cand['text']),evidence.norm(cand['sentence'])):
                return dict(valid=True,choice=key,selected_role=cand['role'],abstained=False)
        return dict(valid=False,reason='out_of_candidate_contract',text=text)
    except (ValueError,TypeError):return dict(valid=False,reason='malformed_JSON')


def compose(q,decisions):
    applicable=[e for e in q['edits'] if e['kind']=='different_phone_substitution']
    assert len(applicable)==len(decisions)
    if not applicable:return q['normalized_output'],True
    if any(not d['decision']['valid'] for d in decisions):return '',False
    out=q['normalized_output']
    for e,d in reversed(list(zip(applicable,decisions))):
        assert e['edit_id']==d['edit_id']
        lo,hi=e['output_span']
        assert out[lo:hi]==e['modified']
        if d['decision']['selected_role']=='original':out=out[:lo]+e['original']+out[hi:]
    return out,True


def prepare():
    assert not (HERE/'protocol.json').exists()
    ec=evidence.load(EVID/'protocol.json');seal=evidence.load(EVID/'evidence_seal.json')
    assert evidence.sha(EVID/'protocol.json')==seal['protocol_sha256']
    for mode in ('true','zero','mismatch'):
        assert evidence.sha(EVID/(mode+'.evidence.jsonl'))==seal['evidence_sha256'][mode]
    manifest=evidence.rows(EVID/'input_edits.jsonl');baseline=evidence.rows(ec['old_predictions'])
    conditions={m:evidence.rows(EVID/(m+'.evidence.jsonl')) for m in ('true','zero','mismatch')}
    assert len(manifest)==len(baseline)==1334
    queries=[]
    for i,(q,r) in enumerate(zip(manifest,baseline)):
        assert q['id']==r['id']
        for j,e in enumerate(q['edits']):
            if e['kind']!='different_phone_substitution':continue
            scores={}
            for m in conditions:
                ee=conditions[m][i]['edits'][j]
                assert ee['edit_id']==e['edit_id'];scores[m]=ee['scores']
            queries.append(make_query(q,e,r['input'],scores))
    assert len(queries)==190 and len({q['edit_id'] for q in queries})==190
    with (HERE/'queries.jsonl').open('x',encoding='utf-8') as stream:
        for q in queries:stream.write(json.dumps(q,ensure_ascii=False)+'\n')
    cp=Path(evidence.load(OLD/'B/selected_checkpoint.json')['checkpoint'])
    meta=evidence.load(cp/'metadata.json');assert (meta['group'],meta['step'],meta['examples'])==('B',2162,17294)
    for name in ('adapter/adapter_config.json','adapter/adapter_model.safetensors'):
        assert evidence.sha(cp/name)==meta['hashes'][name]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
    lengths={}
    for mode in ('none','true','zero','mismatch'):
        ll=[]
        for q in queries:
            text=tok.apply_chat_template(messages(q,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+96<=4096
            ll.append(n)
        lengths[mode]=dict(min=min(ll),max=max(ll),mean=sum(ll)/len(ll))
    p=w.legacy_module(HERE.parent/'staged_coupling_20260917')
    # Baseline replay IDs depend only on order and input lengths, not reference correctness.
    longest=max(range(len(baseline)),key=lambda i:len(json.dumps(p.evidence(baseline[i]),ensure_ascii=False)))
    replay_ids=list(dict.fromkeys([baseline[0]['id'],baseline[1]['id'],baseline[2]['id'],baseline[longest]['id']]))
    deps=[HERE/x for x in ('rereview.py','test_rereview.py','PLAN.md','queries.jsonl')]
    deps += [EVID/'protocol.json',EVID/'input_edits.jsonl',EVID/'evidence_seal.json',EVID/'review.py',
             *[EVID/(m+'.evidence.jsonl') for m in ('true','zero','mismatch')],
             Path(ec['dev']),Path(ec['old_predictions']),cp/'metadata.json',cp/'adapter/adapter_config.json',
             cp/'adapter/adapter_model.safetensors',p.HERE/'run.py',p.ALIGNED,
             HERE.parent/'text_residual_coupling_20260920/diagnostics.py',
             HERE.parent/'text_residual_coupling_20260920/workflow.py',
             BASE/'config.json',BASE/'tokenizer.json',BASE/'tokenizer_config.json',BASE/'chat_template.jinja',
             BASE/'model.safetensors.index.json',*sorted(BASE.glob('*.safetensors'))]
    c=dict(experiment='Frozen9B evidence-conditioned local review without connector',
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),sources={str(x):evidence.sha(x) for x in deps},
        base=str(BASE),checkpoint=str(cp),dev=ec['dev'],baseline=ec['old_predictions'],
        modes=['none','true','zero','mismatch'],n=1334,chars=21102,mentions=599,queries=190,
        reviewed_utterances=len({q['id'] for q in queries}),batch_size=4,max_new_tokens=96,max_sequence=4096,
        seed=917,do_sample=False,dtype='bfloat16',backend='Historical unpatched Transformers text route; no FLA/audio modules',
        no_training=True,no_connector=True,prompt_lengths=lengths,baseline_replay_ids=replay_ids,
        primary='Corpus CER versus frozen old B/text and none; hotword recall, newly wrong/correct repaired errors, paired95%CI',
        controls='All modes same prompt and candidate order; none fields unprovided, others sealed native scores; no reference in model query',
        candidate_order='sha256(917|candidate_order|edit_id) bit, identical across modes; not reference based',
        output_contract='Old JSON text schema: A/B fragment or corresponding full sentence, A/B labels accepted; explicit U/证据不足 keeps draft; malformed/off-contract response scores entire affected utterance empty',
        routing='190 input/output-only different-phone substitution blocks; all1334 outputs evaluated. Other edits untouched. No references or error labels used for routing.',
        scope='Candidate-selection local review, NOT unconstrained second-pass transcription; cannot introduce new words outside candidates or fix untouched positions',
        success='Real evidence must reduce harmful edits while preserve correct repairs and not worsen corpusCER; compare none/erased/mismatch, no checkpoint or threshold selection',
        stopping='Hash/ID/weights/backend replay/batch equivalence/OOM failure stops, preserves artifacts; no protocol reduction or new training',
        limitations=[*ec['limitations'],'Evidence-score task differs from old correction training; a failure can reflect task/format shift',
          'none changes evidence availability; zero/mismatch have OOD score magnitudes, so comparisons do not prove raw auditory understanding',
          'Both native and predefined phonetic scores included; this is evidence-interface pilot, not isolated-score ablation',
          'Abstention retention is an explicit model decision, not score-threshold recovery; invalid responses are NOT copied from draft',
          'Review edits constructed from previously examined dev predictions; retrospective diagnostic, no new heldout claim'])
    evidence.save(HERE/'protocol.json',c)
    evidence.save(HERE/'status.json',dict(status='PREPARED',queries=190,reviewed_utterances=c['reviewed_utterances']))
    print(json.dumps(dict(status='PREPARED',lengths=lengths,reviewed_utterances=c['reviewed_utterances'])),flush=True)


def config():
    c=evidence.load(HERE/'protocol.json')
    for name,expected in c['sources'].items():assert evidence.sha(name)==expected,name
    return c


class Runtime:
    def __init__(self,c):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        set_seed(c['seed']);torch.set_num_threads(6)
        self.tok=AutoTokenizer.from_pretrained(c['base'],trust_remote_code=True)
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        self.tok.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(c['base'],trust_remote_code=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'})
        self.model=PeftModel.from_pretrained(self.model,Path(c['checkpoint'])/'adapter',is_trainable=False)
        self.model.eval();self.model.config.use_cache=True
        for p in self.model.parameters():p.requires_grad_(False)
        saved=load_file(str(Path(c['checkpoint'])/'adapter/adapter_model.safetensors'))
        actual=get_peft_model_state_dict(self.model)
        assert saved.keys()==actual.keys() and all(torch.equal(x,actual[k].cpu()) for k,x in saved.items())
        self.c=c

    def generate(self,queries,mode):
        import torch
        prompts=[self.tok.apply_chat_template(messages(q,mode),tokenize=False,
            add_generation_prompt=True,enable_thinking=False) for q in queries]
        encoded=self.tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt')
        assert encoded.input_ids.shape[1]+self.c['max_new_tokens']<=self.c['max_sequence']
        ids=encoded.input_ids.cuda();mask=encoded.attention_mask.cuda()
        with torch.inference_mode():
            emb=self.model.get_input_embeddings()(ids)
            output=self.model.generate(inputs_embeds=emb,attention_mask=mask,do_sample=False,
                max_new_tokens=self.c['max_new_tokens'],pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,use_cache=True)
        assert output.shape[1]<=self.c['max_new_tokens']
        return [self.tok.decode(x,skip_special_tokens=True) for x in output]


def run():
    c=config();p=w.legacy_module(HERE.parent/'staged_coupling_20260917')
    qs=evidence.rows(HERE/'queries.jsonl');old=evidence.rows(c['baseline'])
    with w.gpu_guard(p):
        rt=Runtime(c)
        # Identity check uses the original text interface, same frozen adapter, no connector object.
        by_id={r['id']:r for r in old};replay=[]
        for uid in c['baseline_replay_ids']:
            r=by_id[uid];actual=p.generate(rt.model,rt.tok,None,r,'B','text')
            assert actual['model_output']==r['model_output'],('Baseline replay differs',uid,actual['model_output'])
            replay.append(dict(id=uid,raw_exact=True))
        batch=rt.generate(qs[:4],'none');single=[rt.generate([q],'none')[0] for q in qs[:4]]
        assert batch==single,('Batch equivalence failed',batch,single)
        evidence.save(HERE/'runtime_audit.json',dict(status='PASS',adapter_reload_exact=True,
            baseline_replay=replay,batch_vs_single_raw_exact=True,batch_queries=[q['edit_id'] for q in qs[:4]],
            batch_raw=batch,trainable_parameters=0,connector_instantiated=False))
        began=time.time()
        for mode in c['modes']:
            final=HERE/(mode+'.decisions.jsonl');assert not final.exists()
            partial=Path(str(final)+'.partial');assert not partial.exists()
            with partial.open('x',encoding='utf-8') as stream:
                for start in range(0,len(qs),c['batch_size']):
                    chunk=qs[start:start+c['batch_size']];rr=rt.generate(chunk,mode)
                    for q,raw in zip(chunk,rr):
                        record=dict(id=q['id'],edit_id=q['edit_id'],mode=mode,model_output=raw,decision=parse(raw,q))
                        stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                    done=start+len(chunk)
                    rec=dict(status='REVIEWING',mode=mode,done=done,total=len(qs),seconds=time.time()-began)
                    evidence.save(HERE/'status.json',rec)
                    if done%20==0 or done==len(qs):print(json.dumps(rec),flush=True)
            assert [r['edit_id'] for r in evidence.rows(partial)]==[q['edit_id'] for q in qs]
            partial.rename(final)
    evidence.save(HERE/'decision_seal.json',dict(protocol_sha256=evidence.sha(HERE/'protocol.json'),
        hashes={m:evidence.sha(HERE/(m+'.decisions.jsonl')) for m in c['modes']},reference_used=False))
    evidence.save(HERE/'status.json',dict(status='REVIEW_COMPLETE_NOT_SCORED'))


def report():
    c=config();seal=evidence.load(HERE/'decision_seal.json');assert not seal['reference_used']
    for m in c['modes']:assert evidence.sha(HERE/(m+'.decisions.jsonl'))==seal['hashes'][m]
    old=evidence.rows(c['baseline']);dev=evidence.rows(c['dev']);manifest=evidence.rows(EVID/'input_edits.jsonl')
    p=w.legacy_module(HERE.parent/'staged_coupling_20260917')
    baseline=p.metrics(old,protected=False);assert (baseline['summary']['errors'],baseline['summary']['chars'],baseline['summary']['hits'],baseline['summary']['mentions'])==(622,21102,469,599)
    metrics={};decision_diagnostics={};hashes={}
    labels={z['edit_id']:z for z in evidence.load(EVID/'posthoc_cases.json')}
    for mode in c['modes']:
        ds=evidence.rows(HERE/(mode+'.decisions.jsonl'));by={}
        for d in ds:by.setdefault(d['id'],[]).append(d)
        predictions=[]
        for q,r,e in zip(manifest,old,dev):
            assert q['id']==r['id']==e['id'] and r['reference']==e['reference']
            text,valid=compose(q,by.get(q['id'],[]))
            if not by.get(q['id']):text=r['prediction']
            predictions.append(dict(id=r['id'],reference=r['reference'],input=r['input'],prediction=text,
                parsed=valid,condition=mode,reviewed_edits=len(by.get(q['id'],[])),audio_id=r['id']))
        target=HERE/(mode+'.predictions.jsonl')
        with target.open('x',encoding='utf-8') as stream:
            for r in predictions:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        assert len(predictions)==1334 and [r['id'] for r in predictions]==[r['id'] for r in old]
        mm=p.metrics(predictions,protected=False);mm['error_diagnostics']=diagnose(predictions,p.evaluator().norm)
        assert (mm['summary']['n'],mm['summary']['chars'],mm['summary']['mentions'])==(1334,21102,599)
        assert sum(mm['error_diagnostics'][k] for k in ('S','D','I'))==mm['summary']['errors']
        metrics[mode]=mm;hashes[mode]=evidence.sha(target);evidence.save(HERE/(mode+'.metrics.json'),mm)
        counts={label:dict(n=0,original_selected=0,modified_selected=0,abstained=0,invalid=0) for label in ('harmful','helpful','neutral')}
        for d in ds:
            label=labels[d['edit_id']]['diagnosis'];cell=counts[label];cell['n']+=1
            dec=d['decision']
            if not dec['valid']:cell['invalid']+=1
            elif dec['abstained']:cell['abstained']+=1
            elif dec['selected_role']=='original':cell['original_selected']+=1
            else:cell['modified_selected']+=1
        changes=dict(changed_utterances=0,better=0,worse=0,equal=0,removed_errors=0,added_errors=0)
        for r,b in zip(predictions,old):
            x=p.evaluator().norm(r['prediction']);y=p.evaluator().norm(b['prediction']);ref=p.evaluator().norm(r['reference'])
            if x==y:continue
            a=evidence.align(ref,x)[0];bb=evidence.align(ref,y)[0]
            changes['changed_utterances']+=1;changes['better']+=a<bb;changes['worse']+=a>bb;changes['equal']+=a==bb
            changes['removed_errors']+=max(0,bb-a);changes['added_errors']+=max(0,a-bb)
        decision_diagnostics[mode]=dict(cells=counts,utterance_changes=changes,
            valid_decisions=sum(d['decision']['valid'] for d in ds),total_decisions=190)
    result=dict(status='COMPLETE',baseline=baseline['summary'],baseline_diagnostics=diagnose(old,p.evaluator().norm),
        summaries={m:mm['summary'] for m,mm in metrics.items()},diagnostics={m:mm['error_diagnostics'] for m,mm in metrics.items()},
        decisions=decision_diagnostics,paired_95ci={m:p.bootstrap(mm['per_utterance'],baseline['per_utterance']) for m,mm in metrics.items()},
        true_minus_controls_95ci={m:p.bootstrap(metrics['true']['per_utterance'],metrics[m]['per_utterance']) for m in ('none','zero','mismatch')},
        decision_hashes=seal['hashes'],prediction_hashes=hashes,protocol_sha256=evidence.sha(HERE/'protocol.json'),
        no_score_threshold_or_restore=True,scope=c['scope'],limitations=c['limitations'])
    evidence.save(HERE/'summary.json',result)
    evidence.save(HERE/'status.json',dict(status='ALL_DONE',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','report','pipeline']);args=ap.parse_args()
    try:
        if args.command=='pipeline':prepare();run();report()
        else:globals()[args.command]()
    except BaseException:
        failure=dict(command=args.command,error=traceback.format_exc(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        evidence.save(HERE/('failure_'+str(time.time_ns())+'.json'),failure)
        evidence.save(HERE/'status.json',dict(status='FAILED',**failure));raise
