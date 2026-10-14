"""First-four-input numerical diagnostic ONLY; no decisions/evaluation/retry."""
import math
import time
import torch
from likelihood import HERE, Scorer, config, previous, rows, save, target_positions

c=config(); r=previous(); p=r.w.legacy_module(HERE.parent/'staged_coupling_20260917')
qs=rows(HERE/'queries.jsonl')[:4]
jobs=[(q,'none',key) for q in qs for key in ('A','B')]
with r.w.gpu_guard(p):
    scorer=Scorer(c)
    batch=scorer.score(jobs);repeat=scorer.score(jobs)
    singles=[scorer.score([job])[0] for job in jobs]
    full=scorer.score(jobs[:1],all_logits=True)[0]
    enc=scorer.encode(jobs);n=max(len(x['ids']) for x in enc)
    padded=[]
    # Same left padded sequence length as batch8, but one sequence per call.
    with torch.inference_mode():
        for x in enc:
            pad=n-len(x['ids'])
            ids=torch.tensor([[scorer.rt.tok.pad_token_id]*pad+x['ids']],device='cuda')
            mask=torch.tensor([[0]*pad+[1]*len(x['ids'])],device='cuda')
            pos=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
            t=len(x['target_ids']);k=t+1
            ll=scorer.rt.model(inputs_embeds=scorer.rt.model.get_input_embeddings()(ids),attention_mask=mask,
                position_ids=pos,use_cache=False,logits_to_keep=k).logits
            logits=ll[0,target_positions(t,k)].float();want=torch.tensor(x['target_ids'],device='cuda')
            lp=(logits.gather(1,want[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)).cpu().tolist()
            padded.append(dict(logp=math.fsum(lp),token_logp=lp))
    def comparison(a,b):
        return dict(max_token_difference=max(abs(x-y) for aa,bb in zip(a,b) for x,y in zip(aa['token_logp'],bb['token_logp'])),
            pair_gap_difference=[(a[i]['logp']-a[i+1]['logp'])-(b[i]['logp']-b[i+1]['logp']) for i in range(0,8,2)],
            winners_equal=[(a[i]['logp']>=a[i+1]['logp'])==(b[i]['logp']>=b[i+1]['logp']) for i in range(0,8,2)])
    result=dict(status='DIAGNOSTIC_ONLY',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        reviewed_first4_inputs_only=True,no_decisions_or_predictions_written=True,reference_not_used=True,
        source_protocol_not_changed=True,batch_repeat=comparison(batch,repeat),batch_single=comparison(batch,singles),
        batch_padded_single=comparison(batch,padded),padded_single_unpadded_single=comparison(padded,singles),
        trimmed_vs_full_first_token_max_difference=max(abs(a-b) for a,b in zip(full['token_logp'],singles[0]['token_logp'])),
        batch_scores=batch,single_scores=singles,sequence_lengths=[len(x['ids']) for x in enc])
    save(HERE/'numerical_diagnostic.json',result)
    print({k:v for k,v in result.items() if k not in ('batch_scores','single_scores')},flush=True)
