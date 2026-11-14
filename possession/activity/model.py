import argparse,json


from pathlib import Path


from functools import lru_cache


import numpy as np


from possession.activity.features import features,reduce_blocks,NAMES


from possession.preprocess import HALVES


from possession.kinematics import decode_general


def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-40,40)))


def train_logistic(x,y,l2=.05,iterations=300):
    """Balanced deterministic logistic regression; y must come from organiser seeds."""
    mean=x.mean(0);scale=np.maximum(x.std(0),.1);z=(x-mean)/scale
    n0=int((y==0).sum());n1=int((y==1).sum())
    if min(n0,n1)<20:raise ValueError('Not enough physical pseudo-labels for both classes')
    weight=np.where(y==1,.5/n1,.5/n0);w=np.zeros(x.shape[1]);intercept=0.
    for _ in range(iterations):
        residual=(sigmoid(z@w+intercept)-y)*weight
        w-=.2*(z.T@residual+l2*w);intercept-=.2*residual.sum()
    return dict(mean=mean,scale=scale,weight=w,intercept=intercept,live_seeds=n0,dead_seeds=n1)


def physical_seeds(x):
    f={name:x[:,i] for i,name in enumerate(NAMES)}
    live=(f['ball_motion']>.7)&(f['running']>.2)&(f['setpiece']<.1)&(f['outside']<.1)
    dead=((f['setpiece']>.5)&(f['walking']>.65)&((f['still']>.5)|(f['missing']>.7)))|((f['outside']>.7)&(f['walking']>.5))
    chosen=live|dead
    assert not (live&dead).any()
    return chosen,dead[chosen].astype(float)


@lru_cache(maxsize=8)
def learned_parameters(l2=.05):
    x=np.vstack(reduce_blocks(features()['x']));chosen,y=physical_seeds(x)
    return train_logistic(x[chosen],y,l2=l2)


def evidence(x,c):
    f={name:x[:,i] for i,name in enumerate(NAMES)}
    if c.get('family','physics')=='seeded_logistic':
        p=learned_parameters(c.get('l2',.05))
        score=((x-p['mean'])/p['scale'])@p['weight']+p['intercept']+c.get('bias',0)
    else:
        score=np.full(len(x),c.get('bias',-.8))
        score+=c.get('missing',2.5)*f['long_gap']+c.get('still',2)*f['still']+c.get('outside',3)*f['outside']
        score+=c.get('setup',1.5)*f['setpiece']*f['walking']+c.get('walking',.5)*f['walking']
        score-=c.get('moving',2)*f['ball_motion']+c.get('running',.5)*f['running']+c.get('pressure',.5)*f['pressure']
        score-=c.get('contact',.5)*f['contact']*(1-f['setpiece'])+c.get('phase',.2)*f['phase']
        score-=c.get('broadcast',.3)*f['broadcast_seen']*(1-f['setpiece'])
        score+=c.get('prepare',0)*f['preparation']-c.get('restart',0)*f['after_restart']
    if c.get('trust',False):score*=f['quality']
    return score


def moving_average(x,radius):
    if not radius:return x.copy()
    ix=np.arange(len(x));lo=np.maximum(0,ix-radius);hi=np.minimum(len(x),ix+radius+1)
    cs=np.r_[0,np.cumsum(x)];return (cs[hi]-cs[lo])/(hi-lo)


def binary_decode(score,enter=5,leave=5,mode='hmm'):
    if mode=='threshold':return (score>0).astype('i1')
    if mode=='hysteresis':
        state=0;out=np.zeros(len(score),dtype='i1');count=0
        for t,value in enumerate(score):
            evidence_for_change=value>.5 if state==0 else value<-.5
            count=count+1 if evidence_for_change else 0
            if count>=(max(1,round(enter)) if state==0 else max(1,round(leave))):state=1-state;count=0
            out[t]=state
        return out
    trans=np.array([[0.,-enter],[-leave,0.]])
    trans-=np.log(np.exp(trans).sum(1))[:,None]
    return decode_general(np.column_stack([np.zeros(len(score)),score]),trans,np.array([0,1]))


def predict(config,details=False):
    data=features();out=np.full(len(data['x']),'X',dtype='<U1');score=np.zeros(len(out))
    for (a,b),x in zip(HALVES,reduce_blocks(data['x'])):
        evidence_score=moving_average(evidence(x,config),config.get('window',3))
        states=binary_decode(evidence_score,config.get('enter',5),config.get('leave',5),config.get('decoder','hmm'))
        out[a:b]=np.repeat(np.array(['L','D'])[states],5)[:b-a]
        score[a:b]=np.repeat(sigmoid(evidence_score),5)[:b-a]
    return (out,score) if details else out


def write(path,labels,score):
    with Path(path).open('w') as file:
        file.write('frame,state,dead_score\n')
        for frame,(label,value) in enumerate(zip(labels,score)):
            file.write(f'{frame},{label},{value:.6f}\n')


def freeze_ownership():
    """Fixed ownership path composed with the activity prediction."""
    from possession.base import predict as predict_parent
    from possession.ownership import strategy_emissions,reduce_emissions
    c=json.loads(Path(__file__).with_name('parent_config.json').read_text())
    labels=predict_parent(c)
    parent=np.full(len(labels),3,dtype='i1')
    for i,label in enumerate('HAD'):parent[labels==label]=i
    owner=parent.copy();e,_,_,_=strategy_emissions(c);switch=c.get('switch',5)
    trans=np.array([[0.,-switch],[-switch,0.]])
    trans-=np.log(np.exp(trans).sum(1))[:,None]
    for (a,b),red in zip(HALVES,reduce_emissions(e,5,c.get('window',5))):
        labels=np.repeat(decode_general(red[:,:2],trans,np.array([0,1])),5)[:b-a]
        missing=owner[a:b]==2;owner[a:b][missing]=labels[missing]
    return parent,owner


def compose(labels,owner):
    result=owner.copy();result[labels=='D']=2;result[labels=='X']=3;return result

