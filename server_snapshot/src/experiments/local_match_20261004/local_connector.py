"""Phonetic-query local compatibility bottleneck, not an ASR decoder."""
import math
import torch
from torch import nn
from local_inputs import remap_span

class LocalMatchConnector(nn.Module):
    def __init__(self,vocabulary_size,state_vectors):
        super().__init__()
        self.query=nn.Embedding(vocabulary_size,128)
        self.norm=nn.LayerNorm(512)
        self.key=nn.Linear(512,128,bias=False);self.value=nn.Linear(512,128,bias=False)
        self.matcher=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Linear(64,1))
        self.states=nn.Parameter(state_vectors.detach().float().clone())
        assert self.states.shape[0]==2

    def forward(self,frames,record,phone_ids,neutral=False):
        assert frames.ndim==2 and frames.shape[1]==512 and len(frames)>0
        frames=self.norm(frames.float());keys=self.key(frames);values=self.value(frames)
        vectors=[];logits=[]
        for slot in record['slots']:
            start,end=remap_span(slot['span'],record['frames'],len(frames))
            for group in slot['groups']:
                q=self.query.weight[phone_ids[group['phone']]]
                weights=(keys[start:end]@q/math.sqrt(128)).softmax(0)
                context=weights@values[start:end]
                features=torch.cat([q,context,q*context,q-context])
                score=self.matcher(features).squeeze(-1);logits.append(score)
                support=score.sigmoid() if not neutral else score*0+.5
                # Both losses and the decoder use THIS compatibility value.
                states=.5*self.states/self.states.norm(dim=-1,keepdim=True).clamp_min(.1)
                vectors.append((1-support)*states[0]+support*states[1])
        if not vectors:
            return self.states.new_zeros((0,self.states.shape[1])),self.states.new_zeros((0,))
        return torch.stack(vectors),torch.stack(logits)

def matching_loss(logits,labels):
    labels=torch.tensor(labels,device=logits.device)
    assert labels.shape==logits.shape
    valid=labels>=0
    if not valid.any():return logits.sum()*0
    values=torch.nn.functional.binary_cross_entropy_with_logits(logits[valid],labels[valid].float(),reduction='none')
    y=labels[valid];positive=values[y==1];negative=values[y==0]
    if len(positive) and len(negative):return .5*(positive.mean()+negative.mean())
    return values.mean()
