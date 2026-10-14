"""Reference-free prompts and explicit train-only candidate labels."""
import copy
import hashlib
import json
import math


def swap(q):
    out=copy.deepcopy(q);out['A'],out['B']=out['B'],out['A'];return out


def render(q,mode,task='full'):
    assert mode in ('none','true','zero','mismatch') and task in ('full','local')
    # Never serialize the source dictionary: explicit whitelist only.
    item={'候选':{}}
    if task=='full':
        item.update(识别证据=q['input_evidence'],初稿=q['draft'],片段位置=q['output_span'])
    for k in ('A','B'):
        cand=q[k];v={'片段':cand['text'],'字典发音':cand['pinyin']}
        if task=='full':v['完整候选句']=cand['sentence']
        ev={}
        for m,n in (('native','字符CTC'),('phonetic_marginal','发音CTC')):
            value=None if mode=='none' else q['evidence'][mode][m][cand['role']]['score']
            ev[n]='未提供' if mode=='none' else (round(value,4) if value is not None and math.isfinite(value) else '不可评分')
        v['声学证据']=ev;item['候选'][k]=v
    instruction=('选择指定片段更有依据的转写，只回答A或B。候选顺序没有优先级。'
        '声学字段是SenseVoice对仅在此片段不同的两个整句候选计算的CTC对数似然，较大表示支持较强，'
        '但包含识别器和书写偏好，不能机械按分数大小选；字典发音可能有多音字问题。'
        '语言通顺或热词存在也不能证明声音说了该词。')
    if task=='local':
        instruction+='本题隐藏上下文，只依据局部候选和可用声学证据判断；声学分数仍是配对整句分数，不是音频片段分数。'
    return [{'role':'system','content':'你是中文语音纠错的候选决策助手。只选择A或B，不解释、不输出JSON或整句。'},
            {'role':'user','content':instruction+'\n'+json.dumps(item,ensure_ascii=False)}]


def preference(q,reference,normalize,distance,split):
    if split!='train':raise ValueError('Only train reference can form training labels')
    err={k:distance(normalize(reference),normalize(q[k]['sentence'])) for k in ('A','B')}
    prob_a=1. if err['A']<err['B'] else 0. if err['A']>err['B'] else .5
    return {'edit_id':q['edit_id'],'prob_A':prob_a,'errors':err,'neutral':err['A']==err['B']}


def label_for(q,label,reverse=False):
    assert q['edit_id']==label['edit_id']
    return 1.-label['prob_A'] if reverse else label['prob_A']


def choose(a,b):
    if not math.isfinite(a) or not math.isfinite(b):raise ValueError('nonfinite choice score')
    return 'A' if a>=b else 'B'


def order_bit(edit_id):
    return int(hashlib.sha256(('917|train_choice|'+edit_id).encode()).hexdigest()[:16],16)%2
