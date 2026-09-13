"""Token-region attribution for missed better candidates; post-hoc dev diagnostic."""
import hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/score-diagnostic-20260913/score_diagnostic_20260913'

def regions(tok, prompt, text, model, device):
    import torch
    completion=json.dumps({'text':text},ensure_ascii=False)
    p=tok(prompt,add_special_tokens=True)['input_ids']
    ids=tok(prompt+completion,add_special_tokens=True)['input_ids']
    if ids[:len(p)]!=p: raise ValueError('token prefix mismatch')
    ids=ids+[tok.eos_token_id]
    x=torch.tensor([ids],device=device)
    with torch.inference_mode():
        z=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits
    lp=(z[0,len(p)-1:-1,:].float().log_softmax(-1).gather(1,x[0,len(p):,None]).squeeze(1)).tolist()
    ct=ids[len(p):-1]
    n=min(len(ct),len(lp))
    return ct[:n],lp[:n]

def split_regions(a, b):
    pre=0
    while pre<min(len(a),len(b)) and a[pre]==b[pre]: pre+=1
    suf=0
    while suf<min(len(a)-pre,len(b)-pre) and a[-1-suf]==b[-1-suf]: suf+=1
    return dict(prefix=[0,pre],middle_a=[pre,len(a)-suf],middle_b=[pre,len(b)-suf],
                suffix_a=[len(a)-suf,len(a)],suffix_b=[len(b)-suf,len(b)],suffix_len=suf)

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    import sys
    base=Path('/root/autodl-tmp/src/logs/hotword_lora_9b_20260908')
    sys.path.insert(0,str(base))
    from workflow import norm
    model_path=Path('/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B')
    adapter=base/'scheduled/high_lr3e-5/adapter/checkpoint-625'
    source=base/'score_diagnostic_20260913/scores.jsonl'
    predictions=base/'scheduled/high_lr3e-5/full_625.predictions.jsonl'
    out=base/'score_diagnostic_20260913/attribution'
    out.mkdir(exist_ok=False)
    rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    pred={r['id']:r for r in [json.loads(s) for s in predictions.read_text().splitlines() if s.strip()]}
    tok=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='auto')
    model=PeftModel.from_pretrained(model,adapter).eval();device=next(model.parameters()).device
    detail=[];started=time.time();total=0
    for idx,r in enumerate(rows):
        cs=r['candidates'];es=r['diagnostic_edits'];orig=cs[0]
        better=[i for i in range(1,len(cs)) if es[i]<es[0] and cs[i]['sum_logprob']<=cs[0]['sum_logprob']]
        if not better: continue
        rr=pred[r['id']]; prompt=tok.apply_chat_template(rr['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        orig_ids,orig_lp=regions(tok,prompt,orig['text'],model,device)
        for i in better:
            alt=cs[i];alt_ids,alt_lp=regions(tok,prompt,alt['text'],model,device)
            rg=split_regions(orig_ids,alt_ids)
            def sm(x,a,b): return float(sum(x[a:b]))
            detail.append(dict(id=r['id'],original=orig['text'],alternative=alt['text'],
                original_error=es[0],alternative_error=es[i],token_counts=[len(orig_ids),len(alt_ids)],
                total_margin=alt['sum_logprob']-orig['sum_logprob'],mean_margin=alt['mean_logprob']-orig['mean_logprob'],
                original_regions=dict(prefix=sm(orig_lp,*rg['prefix']),middle=sm(orig_lp,*rg['middle_a']),suffix=sm(orig_lp,*rg['suffix_a'])),
                alternative_regions=dict(prefix=sm(alt_lp,*rg['prefix']),middle=sm(alt_lp,*rg['middle_b']),suffix=sm(alt_lp,*rg['suffix_b'])),
                region_bounds=rg))
            total+=2
        if (idx+1)%20==0: print(json.dumps(dict(rows=idx+1,pairs=len(detail),forwards=total,elapsed=time.time()-started)),flush=True)
    (out/'pairs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in detail),encoding='utf-8')
    summary=dict(classification='post-hoc dev diagnostic; reference used only to identify missed better candidates',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pairs=len(detail),forwards=total,elapsed_seconds=time.time()-started,
        note='Regions are token-aligned common prefix/middle/suffix; middle often includes JSON boundary effects and must not be called entity-only without text alignment')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
