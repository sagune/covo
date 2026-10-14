"""Input-only character-slot candidates; TRAIN reference labels are separate."""
import math
import unicodedata

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC',str(s)) if not c.isspace() and not unicodedata.category(c).startswith('P')).strip()

def aligned_characters(base,other):
    n,m=len(base),len(other); d=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+(base[i-1]!=other[j-1]),d[i-1][j]+1,d[i][j-1]+1)
    result=[None]*n;i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(base[i-1]!=other[j-1]):result[i-1]=j-1;i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:i-=1
        else:j-=1
    return result

def pronunciation(text):
    from pypinyin import lazy_pinyin,Style
    result=lazy_pinyin(text,style=Style.NORMAL,errors=lambda s:list(s))
    assert len(result)==len(text),(text,result)
    return result

def candidates(row,vocabulary):
    base=norm(row['input']['asr_top1']); assert base,row['id']
    texts=list(dict.fromkeys([base,*[norm(x) for x in row['input'].get('nbest',[])[:8]]]))
    known=set(vocabulary[1:]); slots=[dict(position=i,base=c,groups=[]) for i,c in enumerate(base)]
    for text in texts:
        phones=pronunciation(text)
        for i,j in enumerate(aligned_characters(base,text)):
            if j is None or phones[j] not in known:continue
            groups=slots[i]['groups'];match=next((g for g in groups if g['phone']==phones[j]),None)
            if match:
                if text[j] not in match['aliases']:match['aliases'].append(text[j])
            elif len(groups)<4:groups.append(dict(phone=phones[j],aliases=[text[j]]))
    return dict(id=row['id'],text=base,slots=slots)

def label_slots(row,record):
    assert row['split']=='train','Reference labels are TRAIN-only'
    ref=norm(row['reference']); phones=pronunciation(ref); alignment=aligned_characters(record['text'],ref)
    labels=[]
    for slot,j in zip(record['slots'],alignment):
        if j is None:
            labels.extend([-1]*len(slot['groups']));continue
        p=phones[j]
        # Latin/numerals and dictionary unknowns are explicitly unlabelled.
        valid=p.isascii() and p.isalpha()
        labels.extend([int(g['phone']==p) if valid else -1 for g in slot['groups']])
    return labels

def timing(record,frames,phone_logits,vocabulary,midpoint_fn):
    phones=pronunciation(record['text']); ids={p:i for i,p in enumerate(vocabulary)}
    positions=[i for i,p in enumerate(phones) if p in ids and ids[p]>0]
    fallback=None
    if positions:
        try:mids=midpoint_fn(phone_logits.log_softmax(-1),[ids[phones[i]] for i in positions])
        except AssertionError:
            mids=[(i+.5)*frames/len(positions) for i in range(len(positions))];fallback='infeasible_input_ctc_uniform'
        borders=[0]+[max(0,min(frames,int((a+b)/2))) for a,b in zip(mids,mids[1:])]+[frames]
        spans={i:[min(borders[k],frames-1),max(min(borders[k],frames-1)+1,borders[k+1])] for k,i in enumerate(positions)}
    else:spans={};fallback='no_known_input_phones'
    for slot in record['slots']:
        slot['span']=spans.get(slot['position'],[0,frames])
        slot['timing_anchored']=slot['position'] in spans
    record.update(frames=frames,timing_fallback=fallback)
    return record

def remap_span(span,source_frames,target_frames):
    a,b=span; start=min(target_frames-1,math.floor(a*target_frames/source_frames))
    return start,max(start+1,min(target_frames,math.ceil(b*target_frames/source_frames)))
