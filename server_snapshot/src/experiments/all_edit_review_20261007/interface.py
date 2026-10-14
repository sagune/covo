"""All-edit contextual A/B review; no reference fields in model input."""
import copy
import hashlib
import json
import math


def swap(q):
    out = copy.deepcopy(q)
    out['A'], out['B'] = out['B'], out['A']
    return out


def render(q, mode, task='full'):
    assert task == 'full' and mode in ('none', 'true', 'zero', 'mismatch')
    item = {'识别证据': q['input_evidence'], '初稿': q['draft'],
            '指定片段位置': q['output_span'], '候选': {}}
    for key in ('A', 'B'):
        cand = q[key]
        ev = {}
        for method, name in (('native', '字符CTC'), ('phonetic_marginal', '发音CTC')):
            value = None if mode == 'none' else q['evidence'][mode][method][cand['role']]['score']
            ev[name] = '未提供' if mode == 'none' else (round(value, 4) if value is not None and math.isfinite(value) else '不可评分')
        item['候选'][key] = {'片段': cand['text'], '字典发音': cand['pinyin'],
                            '完整候选句': cand['sentence'], '声学证据': ev}
    instruction = (
        '选择指定片段更有依据的转写，只回答A或B。候选顺序和初稿都没有优先级。'
        '两候选可能同音、异音，也可能有增删字；空片段表示不保留该位置的字。'
        '综合完整上下文、N-best、热词候选与可用声学证据。热词表不表示该词一定被说出，不能一律恢复热词或原识别。'
        '同音字尤其是姓名、机构、地名，需要结合上下文和提供的实体候选判断书写，不能要求发音分数区别同音汉字。'
        '对不同发音及增删字，声学证据可辅助判断内容是否被说出。'
        'CTC字段是SenseVoice对仅在指定片段不同的两个完整候选句计算的对数似然，较大支持较强；'
        '字符CTC含识别器书写偏好，发音CTC依赖可能错误的字典读音，均不是校准置信度，不能机械按分数大小选择。'
        '没有声学证据时不能编造，不能仅因为文字通顺而忽略识别证据。只选择本次候选，不自由改写。')
    return [{'role': 'system', 'content': '你是中文语音纠错的候选决策助手。只选择A或B，不解释，不输出JSON或整句。'},
            {'role': 'user', 'content': instruction + '\n' + json.dumps(item, ensure_ascii=False)}]


def preference(q, reference, normalize, distance, split):
    if split != 'train':
        raise ValueError('Only train reference can form labels')
    err = {k: distance(normalize(reference), normalize(q[k]['sentence'])) for k in ('A', 'B')}
    prob = 1. if err['A'] < err['B'] else 0. if err['A'] > err['B'] else .5
    return {'edit_id': q['edit_id'], 'prob_A': prob, 'errors': err, 'neutral': err['A'] == err['B']}


def label_for(q, label, reverse=False):
    assert q['edit_id'] == label['edit_id']
    return 1. - label['prob_A'] if reverse else label['prob_A']


def choose(a, b):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('Nonfinite choice logits')
    return 'A' if a >= b else 'B'


def order_bit(edit_id):
    return int(hashlib.sha256(('917|train_choice|' + edit_id).encode()).hexdigest()[:16], 16) % 2
