"""Frozen-adapter, reference-blind candidate scoring vs cached free generation."""
import copy
import difflib
import hashlib
import json
import time
from pathlib import Path


def candidates(inp, prediction):
    """Full sentence candidates and one-edit reversions; no reference access."""
    cb=inp['asr_top1']
    out=[prediction,cb]
    for tag,i,j,k,l in difflib.SequenceMatcher(None,prediction,cb,autojunk=False).get_opcodes():
        if tag!='equal':out.append(prediction[:i]+cb[k:l]+prediction[j:])
    return list(dict.fromkeys(out))


def main():
    import sys
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    base=Path('/root/autodl-tmp/src/logs/hotword_lora_9b_20260908')
    sys.path.insert(0,str(base))
    from workflow import evaluate,norm,edit_distance
    model_path=Path('/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B')
    adapter=base/'scheduled/high_lr3e-5/adapter/checkpoint-625'
    source=base/'scheduled/high_lr3e-5/full_625.predictions.jsonl'
    out=base/'score_diagnostic_20260913'
    out.mkdir(exist_ok=False)
    def save(name,obj):
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    started=time.time()
    try:
        rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        assert len(rows)==1334 and len({r['id'] for r in rows})==1334
        assert all(r['split']=='dev' and not r.get('parse_warnings') for r in rows)
        inputs={r['id']:r for r in [json.loads(s) for s in (base/'dev.unified.jsonl').read_text().splitlines() if s.strip()]}
        assert all(r['messages']==inputs[r['id']]['messages'] for r in rows)
        save('PROTOCOL.json',dict(classification='full dev diagnostic, already used for checkpoint selection',
            source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            model=str(model_path),adapter=str(adapter),samples=len(rows),
            candidates='cached prediction, CB top1, each individual char-aligned edit reverted to CB; no gold',
            selection='Maximum sum log probability of canonical JSON completion plus EOS; ties retain original',
            caveat='Canonical JSON serialization may differ from cached raw generation; not proof of cognitive ability',
            normalization='workflow.evaluate: OpenCC t2s, NFKC, punctuation/space removal',
            objectives='CER nonincrease AND designated hotword recall increase versus cached best',
            total_candidates=sum(len(candidates(r['input'],r['prediction'])) for r in rows)))
        tok=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='auto')
        model=PeftModel.from_pretrained(model,adapter).eval()
        device=next(model.parameters()).device
        scored=0
        def score(prompt,text):
            completion=json.dumps({'text':text},ensure_ascii=False)
            ids=tok(prompt+completion,add_special_tokens=True)['input_ids']
            prefix=tok(prompt,add_special_tokens=True)['input_ids']
            if ids[:len(prefix)]!=prefix:raise ValueError('Prompt/completion token boundary mismatch')
            ids=ids+[tok.eos_token_id]
            tensor=torch.tensor([ids],device=device)
            with torch.inference_mode():
                logits=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False).logits
                logits=logits[0,len(prefix)-1:-1,:].float()
                labels=tensor[0,len(prefix):]
                lp=logits.gather(1,labels[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)
                return dict(sum_logprob=float(lp.sum()),mean_logprob=float(lp.mean()),completion_tokens=len(labels))
        with (out/'scores.jsonl').open('x',encoding='utf-8') as detail, (out/'selected.predictions.jsonl').open('x',encoding='utf-8') as selected:
            counts=dict(changed=0,better=0,worse=0,equal=0,lower_cer_candidate_available=0,lower_cer_candidate_scored_above_original=0)
            for index,r in enumerate(rows):
                texts=candidates(r['input'],r['prediction'])
                prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                scores=[dict(text=t,**score(prompt,t)) for t in texts]
                scored+=len(scores)
                winner=max(range(len(scores)),key=lambda i:scores[i]['sum_logprob'])
                record=copy.deepcopy(r);record['prediction']=texts[winner];record['parse_warnings']=[]
                selected.write(json.dumps(record,ensure_ascii=False)+'\n');selected.flush()
                # References used only below this point for post-selection diagnostics.
                gold=norm(r['reference'])
                errors=[edit_distance(list(gold),list(norm(t))) for t in texts]
                counts['changed']+=winner!=0;counts['better']+=errors[winner]<errors[0]
                counts['worse']+=errors[winner]>errors[0];counts['equal']+=errors[winner]==errors[0]
                counts['lower_cer_candidate_available']+=min(errors)<errors[0]
                counts['lower_cer_candidate_scored_above_original']+=any(e<errors[0] and s['sum_logprob']>scores[0]['sum_logprob'] for e,s in zip(errors,scores))
                detail.write(json.dumps(dict(id=r['id'],selected=winner,candidates=scores,diagnostic_edits=errors),ensure_ascii=False)+'\n');detail.flush()
                save('STATUS.json',dict(stage='scoring',samples=index+1,total=len(rows),candidates=scored,elapsed_seconds=time.time()-started))
                if (index+1)%25==0:print(json.dumps(dict(samples=index+1,candidates=scored,elapsed=time.time()-started)),flush=True)
        baseline=evaluate(source);result=evaluate(out/'selected.predictions.jsonl')
        save('RESULTS.json',dict(baseline=baseline,selected=result,diagnostics=counts,
            joint_objective_met=result['cer']<=baseline['cer'] and result['recall']>baseline['recall']))
        save('DONE.json',dict(samples=len(rows),elapsed_seconds=time.time()-started))
        print('DONE',flush=True)
    except Exception as e:
        save('FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
