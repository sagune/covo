"""CPU-only, reference-aware error diagnosis. Never used as a deployed selector."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from pypinyin import Style, lazy_pinyin, pinyin


def norm(s):
    s = unicodedata.normalize('NFKC', str(s))
    return ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith('P')).strip()


def load(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def align(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0]=i
    for j in range(len(b)+1): d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i][j]=min(d[i-1][j-1]+(a[i-1]!=b[j-1]),d[i-1][j]+1,d[i][j-1]+1)
    i,j=len(a),len(b); ops=[]
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            ops.append(dict(op='M' if a[i-1]==b[j-1] else 'S',ri=i-1,pj=j-1,r=a[i-1],p=b[j-1]));i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            ops.append(dict(op='D',ri=i-1,pj=j,r=a[i-1],p=''));i-=1
        else:
            ops.append(dict(op='I',ri=i,pj=j-1,r='',p=b[j-1]));j-=1
    ops.reverse()
    assert sum(o['op']!='M' for o in ops)==d[-1][-1]
    return d[-1][-1],ops


@lru_cache(maxsize=20000)
def syllables(s):
    out=lazy_pinyin(s,style=Style.TONE3,neutral_tone_with_five=True,errors=lambda x:list(x))
    assert len(out)==len(s),(s,out)
    return tuple(out)


def base(py): return re.sub(r'[1-5]$','',py)
def han(c): return len(c)==1 and '\u4e00'<=c<='\u9fff'


@lru_cache(maxsize=5000)
def all_pron(c):
    return set(pinyin(c,style=Style.TONE3,heteronym=True,neutral_tone_with_five=True)[0]) if han(c) else set()


def phonetic(a,b):
    if not a or not b: return 'missing_or_inserted'
    if not all(han(c) for c in a+b): return 'non_han_or_mixed'
    x,y=syllables(a),syllables(b)
    if x==y: return 'same_syllables_and_tones'
    if tuple(map(base,x))==tuple(map(base,y)): return 'same_syllables_different_tones'
    if len(a)==len(b) and all(all_pron(x)&all_pron(y) for x,y in zip(a,b)):
        return 'possible_homophone_alternate_reading'
    return 'different_syllables'


def texts(v): return [norm(t if isinstance(t,str) else t.get('text','')) for t in v or []]


def blocks(ops):
    out=[]; current=[]
    for o in ops:
        if o['op']=='M':
            if current: out.append(current); current=[]
        else: current.append(o)
    if current: out.append(current)
    return out


def spans(ref,terms):
    out=[]
    for t in terms:
        start=0
        while t and (idx:=ref.find(t,start))>=0:
            out.append((idx,idx+len(t),t)); start=idx+1
    return out


def overlaps(o,ss):
    return any(a<o['ri']<b if o['op']=='I' else a<=o['ri']<b for a,b,_ in ss)


def aligned_piece(ops,start,end):
    return ''.join(o['p'] for o in ops if start<=o['ri']<end and o['op']!='D')


def main(args):
    root=Path(args.root); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    paths={'original':root/'D/dev/text.predictions.jsonl','joint_audio':root/'B/dev/true.predictions.jsonl',
           'joint_text':root/'B/dev/text.predictions.jsonl','joint_zero':root/'B/dev/zero.predictions.jsonl'}
    rr={k:load(v) for k,v in paths.items()}; rows=rr['original']; expected=json.loads((root/'summary.json').read_text())
    for other in rr.values():
        assert [(r['id'],r['reference'],r['input']) for r in other]==[(r['id'],r['reference'],r['input']) for r in rows]
    annotated=defaultdict(list)
    for line in Path(args.annotations).read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('\t')
        if len(fields)>=2: annotated[fields[1].strip()].append(norm(fields[0]))
    summary=Counter(); sd=Counter(); ph=Counter(); ph_context=Counter(); origins=Counter(); coverage=Counter()
    pairs=Counter(); pair_examples=defaultdict(list); error_rows=[]; edit_rows=[]; missing_mentions=[]
    control=defaultdict(Counter); candidate_oracle_errors=0
    for rownum,row in enumerate(rows):
        uid=row['id']; ref=norm(row['reference']); pred=norm(row['prediction']); inp=norm(row['input']['asr_top1'])
        ie,io=align(ref,inp); oe,oo=align(ref,pred)
        summary['rows']+=1; summary['chars']+=len(ref); summary['input_errors']+=ie; summary['output_errors']+=oe
        summary['input_correct']+=ie==0; summary['output_correct']+=oe==0
        summary['input_correct_broken']+=ie==0 and oe>0
        summary['errors_on_input_correct']+=oe if ie==0 else 0
        summary['input_wrong_fully_fixed']+=ie>0 and oe==0
        summary['input_wrong_still_wrong']+=ie>0 and oe>0
        summary['improved']+=oe<ie; summary['worsened']+=oe>ie; summary['same_error_count']+=oe==ie
        if oe: summary['wrong_output_rows']+=1
        for k,other in rr.items():
            err,_=align(ref,norm(other[rownum]['prediction']))
            if oe:
                control[k]['baseline_wrong_rows']+=1
                control[k]['fully_fixed_baseline_wrong']+=err==0
                control[k]['improved_baseline_wrong']+=err<oe
                control[k]['worsened_baseline_wrong']+=err>oe
                control[k]['errors_on_baseline_wrong']+=err
            else:
                control[k]['broken_baseline_correct']+=err>0
                control[k]['errors_on_baseline_correct']+=err
        # Only prompt-visible top1 and at most8 hypotheses, not the hidden16-candidate pool.
        candidates=list(dict.fromkeys([inp]+texts(row['input'].get('nbest'))[:8]))
        cs=[align(ref,c) for c in candidates]; best=min(e for e,_ in cs); candidate_oracle_errors+=best
        if oe:
            coverage['wrong_rows_ref_exact_in_visible_pool']+=ref in candidates
            coverage['wrong_rows_visible_pool_has_lower_cer_candidate']+=best<oe
            coverage['wrong_rows_no_candidate_better_than_output']+=best>=oe
        terms=annotated[uid]; ss=spans(ref,terms)
        visible_hotwords=(texts(row['input'].get('prompt_hotwords')) or texts(row['input'].get('hotwords')))[:32]
        hs=spans(ref,visible_hotwords)
        initial_errors=[o for o in io if o['op']!='M']
        initial_signatures={(o['op'],o['ri'],o['p']) for o in initial_errors}
        input_bad_ref={o['ri'] for o in initial_errors if o['op']!='I'}
        input_bad_gaps={o['ri'] for o in initial_errors if o['op']=='I'}
        refpy,predpy=syllables(ref),syllables(pred)
        for o in oo:
            if o['op']=='M': continue
            sd[o['op']]+=1
            if (o['op'],o['ri'],o['p']) in initial_signatures: origin='same_input_error_retained'
            elif o['ri'] in (input_bad_gaps if o['op']=='I' else input_bad_ref): origin='changed_preexisting_error'
            else: origin='error_at_previously_correct_position'
            origins[origin]+=1
            known=any(not any(q['op']!='M' and q['ri']==o['ri'] and (q['op']=='I')==(o['op']=='I') for q in candops) for _,candops in cs)
            coverage['output_error_positions_correct_in_visible_candidate']+=known
            coverage['output_errors_overlapping_designated_hotword']+=overlaps(o,ss)
            coverage['output_errors_overlapping_prompt_hotword']+=overlaps(o,hs)
            if o['op']=='S':
                typ=phonetic(o['r'],o['p']); ph[typ]+=1
                if han(o['r']) and han(o['p']):
                    ctx='same_syllables_and_tones' if refpy[o['ri']]==predpy[o['pj']] else ('same_syllables_different_tones' if base(refpy[o['ri']])==base(predpy[o['pj']]) else 'different_syllables')
                else: ctx='non_han_or_mixed'
                ph_context[ctx]+=1
                pairs[(o['r'],o['p'])]+=1
                if len(pair_examples[(o['r'],o['p'])])<4: pair_examples[(o['r'],o['p'])].append(uid)
            else: typ='deletion' if o['op']=='D' else 'insertion'
            edit_rows.append(dict(id=uid,operation=o['op'],reference_position=o['ri'],prediction_position=o['pj'],
                                  correct=o['r'],wrong=o['p'],phonetic_type=typ,origin=origin,
                                  candidate_correct_at_position=known,designated_hotword_overlap=overlaps(o,ss)))
        for term in terms:
            summary['mentions']+=1; summary['input_hits']+=term in inp; summary['output_hits']+=term in pred
            if term in pred: continue
            present=term in inp
            coverage['lost_input_mentions' if present else 'unrecovered_input_missing_mentions']+=1
            if present:
                coverage['lost_mentions_prompt_contains_term']+=term in visible_hotwords
                coverage['lost_mentions_visible_candidate_contains_term']+=any(term in c for c in candidates)
            pos=ref.find(term); wrong=aligned_piece(oo,pos,pos+len(term)) if pos>=0 else ''
            missing_mentions.append(dict(id=uid,term=term,was_in_input=present,in_prompt=term in visible_hotwords,
                                         in_visible_candidate=any(term in c for c in candidates),output_at_ref_span=wrong,
                                         phonetic_type=phonetic(term,wrong),joint_audio_has_term=term in norm(rr['joint_audio'][rownum]['prediction']),
                                         joint_text_has_term=term in norm(rr['joint_text'][rownum]['prediction'])))
        if oe:
            eb=[]
            for block in blocks(oo):
                right=''.join(o['r'] for o in block); wrong=''.join(o['p'] for o in block)
                eb.append(dict(correct=right,wrong=wrong,phonetic_type=phonetic(right,wrong),
                               operations=''.join(o['op'] for o in block)))
            error_rows.append(dict(id=uid,reference=ref,input=inp,output=pred,input_errors=ie,output_errors=oe,
                                   blocks=eb,reference_exact_in_visible_candidates=ref in candidates,visible_oracle_errors=best,
                                   prompt_hotwords=visible_hotwords,nbest=candidates,
                                   joint_audio=norm(rr['joint_audio'][rownum]['prediction']),joint_text=norm(rr['joint_text'][rownum]['prediction'])))
    assert (summary['rows'],summary['chars'],summary['mentions'],summary['output_errors'],summary['output_hits'])==(1334,21102,599,663,466)
    assert dict(sd)=={'S':584,'D':59,'I':20} or (sd['S'],sd['D'],sd['I'])==(584,59,20)
    assert summary['input_hits']==540
    for k,g in [('original','D_text'),('joint_audio','B_true'),('joint_text','B_text'),('joint_zero','B_zero')]:
        actual=control[k]['errors_on_baseline_wrong']+control[k]['errors_on_baseline_correct']
        assert actual==expected['summaries'][g]['errors']
    result=dict(summary=dict(summary),edit_types=dict(sd),isolated_character_phonetics=dict(ph),sentence_context_phonetics=dict(ph_context),
                alignment_based_error_origins=dict(origins),text_evidence_coverage=dict(coverage),
                visible_candidate_oracle=dict(errors=candidate_oracle_errors,cer=candidate_oracle_errors/summary['chars'],reference_aware=True),
                same_model_controls={k:dict(v) for k,v in control.items()},
                missing_mention_phonetics=dict(Counter(r['phonetic_type'] for r in missing_mentions)),
                lost_mention_phonetics=dict(Counter(r['phonetic_type'] for r in missing_mentions if r['was_in_input'])),
                most_common_substitutions=[dict(correct=a,wrong=b,count=n,phonetic_type=phonetic(a,b),example_ids=pair_examples[(a,b)]) for (a,b),n in pairs.most_common(45)],
                source_hashes={k:sha(v) for k,v in paths.items()},pypinyin_method='0.55.0, TONE3, neutral5; sentence-context default readings plus isolated/alternate-reading checks',
                caveats=['No waveform listening in this pass: phonetic spelling is a proxy, not the observed pronunciation',
                         'Optimal edit alignment tie-break diagonal/deletion/insertion affects per-position origin/support counts',
                         'Designated hotwords are not a complete named-entity annotation',
                         'Candidate oracle and ref-position coverage use references only for diagnosis, not deployable decisions',
                         'Audio controls use the jointly trained model, not identical original30022 weights',
                         'All conclusions concern previously examined1334 contextual dev'])
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,data in [('error_rows',error_rows),('error_edits',edit_rows),('missing_mentions',missing_mentions)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (out/'error_edits.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(edit_rows[0]));writer.writeheader();writer.writerows(edit_rows)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--annotations',required=True);ap.add_argument('--output',required=True)
    main(ap.parse_args())
