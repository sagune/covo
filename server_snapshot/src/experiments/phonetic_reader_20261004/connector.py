"""Time-local soft pronunciation anchors plus continuous acoustic evidence."""
import torch
from torch import nn

def segments(logits, maximum=16):
    """Partition ALL frames by posterior argmax runs, bounded by maximum.

    Boundaries use audio only, never a transcript/reference. Blank runs are kept.
    This is temporal segmentation, not reference-based text forced alignment.
    """
    assert logits.ndim == 2 and len(logits) and maximum>0
    labels=logits.detach().argmax(-1).tolist(); result=[]; start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start] or i-start>=maximum:
            result.append((start,i)); start=i
    assert sum(b-a for a,b in result)==len(labels)
    return result

class PronunciationConnector(nn.Module):
    def __init__(self, phone_head, anchors, hidden_size):
        super().__init__()
        assert anchors.ndim==2 and anchors.shape[1]==hidden_size
        self.phone_head=phone_head.requires_grad_(False)
        self.register_buffer('anchors',anchors.detach().float())
        self.acoustic=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),
                                    nn.Linear(256,hidden_size,bias=False))
        nn.init.zeros_(self.acoustic[-1].weight)
        self.pronunciation_gain=nn.Parameter(torch.tensor(-2.2))

    def forward(self,frames,phone_enabled=True):
        assert frames.ndim==2 and frames.shape[1]==512 and len(frames)
        assert torch.isfinite(frames).all()
        with torch.no_grad():
            logits=self.phone_head(frames.float()); probabilities=logits.softmax(-1)
            spans=segments(logits)
            local_frames=torch.stack([frames[a:b].float().mean(0) for a,b in spans])
            local_phones=torch.stack([probabilities[a:b].mean(0) for a,b in spans])
            # Blank contributes zero. Do not renormalize it away or discard uncertainty.
            semantics=local_phones @ self.anchors
        result=self.acoustic(local_frames)
        if phone_enabled:
            result=result+self.pronunciation_gain.sigmoid()*semantics
        assert torch.isfinite(result).all()
        return result

def forced_phone_midpoints(log_probs, target):
    """CTC Viterbi TRAIN-label alignment, never used to construct inference inputs.

    Dictionary reference phones supply weak training timestamps, not listened gold
    boundaries. Every phone gets one midpoint, including repeated phone targets.
    """
    assert log_probs.ndim==2 and target and all(x>0 for x in target)
    scores=log_probs.detach().cpu().tolist(); states=[0]
    for value in target: states.extend((value,0))
    count=len(states); previous=[float('-inf')]*count
    previous[0]=scores[0][0]; previous[1]=scores[0][states[1]]; backs=[]
    for row in scores[1:]:
        current=[]; back=[]
        for s,value in enumerate(states):
            choices=[s]
            if s: choices.append(s-1)
            if s>=2 and value and value!=states[s-2]: choices.append(s-2)
            source=max(choices,key=lambda j:previous[j])
            current.append(previous[source]+row[value]); back.append(source)
        previous=current; backs.append(back)
    position=max((count-1,count-2),key=lambda s:previous[s])
    assert previous[position]!=float('-inf'), 'Infeasible CTC train-label alignment'
    path=[position]
    for back in reversed(backs):
        position=back[position]; path.append(position)
    path.reverse(); result=[]
    for index in range(len(target)):
        occupied=[t for t,s in enumerate(path) if s==2*index+1]
        assert occupied, 'Missing phone in Viterbi path'
        result.append((occupied[0]+occupied[-1])/2)
    return result
