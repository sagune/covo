"""Before/after patch counterfactual, first four input-selected pairs, no references."""
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'candidate_likelihood_20261005'
sys.path.insert(0,str(OLD))
from likelihood import Scorer, config, previous, rows, save, sha, target_positions
from padding_fix import install


def compare(a,b):
    return dict(max_token_difference=max(abs(x-y) for aa,bb in zip(a,b) for x,y in zip(aa['token_logp'],bb['token_logp'])),
        pair_gap_difference=[(a[i]['logp']-a[i+1]['logp'])-(b[i]['logp']-b[i+1]['logp']) for i in range(0,len(a),2)],
        winners_equal=[(a[i]['logp']>=a[i+1]['logp'])==(b[i]['logp']>=b[i+1]['logp']) for i in range(0,len(a),2)])


def padded_scores(scorer,jobs,n):
    import torch
    result=[]
    with torch.inference_mode():
        for x in scorer.encode(jobs):
            pad=n-len(x['ids']);assert pad>=0
            ids=torch.tensor([[scorer.rt.tok.pad_token_id]*pad+x['ids']],device='cuda')
            mask=torch.tensor([[0]*pad+[1]*len(x['ids'])],device='cuda')
            pos=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
            t=len(x['target_ids']);k=t+1
            ll=scorer.rt.model(inputs_embeds=scorer.rt.model.get_input_embeddings()(ids),attention_mask=mask,
                position_ids=pos,use_cache=False,logits_to_keep=k).logits
            logits=ll[0,target_positions(t,k)].float();wanted=torch.tensor(x['target_ids'],device='cuda')
            lp=(logits.gather(1,wanted[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)).cpu().tolist()
            assert all(math.isfinite(v) for v in lp)
            result.append(dict(logp=math.fsum(lp),token_logp=lp))
    return result


def main():
    assert not (HERE/'maskfix_audit.json').exists()
    c=config();r=previous();p=r.w.legacy_module(OLD.parent/'staged_coupling_20260917')
    # The first4 input-order pairs, fixed before any reference/effect checks.
    qs=rows(OLD/'queries.jsonl')[:4]
    jobs=[(q,'none',k) for q in qs for k in ('A','B')]
    frozen=rows(OLD/'queries.jsonl')
    save(HERE/'protocol.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        no_training=True,no_reference=True,no_formal_predictions=True,first4_inputs=[q['edit_id'] for q in qs],
        source_protocol_sha256=sha(OLD/'protocol.json'),queries_sha256=sha(OLD/'queries.jsonl'),
        source_hashes={x:sha(HERE/x) for x in ('padding_fix.py','test_padding_fix.py','check_maskfix.py','PLAN.md')},
        intervention='Only remove batch>1 guard for padding hidden-state masking in current process',
        evaluation='Exact same jobs and weights before/after; original package unchanged',
        scope='Numerical patch audit only; not an ASR effect evaluation'))
    save(HERE/'status.json',dict(status='CHECKING_MASK_FIX'))
    with r.w.gpu_guard(p):
        scorer=Scorer(c)
        n=max(len(x['ids']) for x in scorer.encode(jobs))
        before_batch=scorer.score(jobs)
        before_single=[scorer.score([j])[0] for j in jobs]
        before_padded=padded_scores(scorer,jobs,n)
        patch=install()
        after_batch=scorer.score(jobs)
        after_single=[scorer.score([j])[0] for j in jobs]
        after_padded=padded_scores(scorer,jobs,n)
        repeated_padded=padded_scores(scorer,jobs,n)
        full=scorer.score(jobs[:1],all_logits=True)[0]
        audit=dict(status='COMPLETE',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),patch=patch,
            before_padded_vs_unpadded=compare(before_padded,before_single),
            after_padded_vs_unpadded=compare(after_padded,after_single),
            before_batch_vs_padded_single=compare(before_batch,before_padded),
            after_batch_vs_padded_single=compare(after_batch,after_padded),
            patch_batch_unchanged=compare(before_batch,after_batch),
            patch_unpadded_single_unchanged=compare(before_single,after_single),
            patched_padded_repeat=compare(after_padded,repeated_padded),
            remaining_batch_vs_unpadded=compare(after_batch,after_single),
            trimmed_vs_full=max(abs(a-b) for a,b in zip(full['token_logp'],after_single[0]['token_logp'])),
            scores=dict(before_batch=before_batch,before_single=before_single,before_padded=before_padded,
                after_batch=after_batch,after_single=after_single,after_padded=after_padded),
            reference_used=False,formal_scores_or_predictions_written=False,queries_total=len(frozen))
        save(HERE/'maskfix_audit.json',audit)
        print({k:v for k,v in audit.items() if k!='scores'},flush=True)
    save(HERE/'status.json',dict(status='AUDIT_DONE_NO_FORMAL_EVAL',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


if __name__=='__main__':
    import traceback
    try:main()
    except BaseException:
        save(HERE/'failure.json',dict(error=traceback.format_exc(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        save(HERE/'status.json',dict(status='FAILED'));raise
