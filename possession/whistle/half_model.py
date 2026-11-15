"""Physical activity evidence and fixed whistle features, decoded per half."""
import argparse
import json
from pathlib import Path

import numpy as np

from .activity_v31 import AudioPossession, half_events, signals
from .fusion import decode
from ..activity.model import train_logistic, physical_seeds, moving_average, evidence
from ..preprocess import HALVES


TAG_NAMES={'setpiece','preparation','after_restart','phase'}
FEATURE_SETS={
    'full':None,
    'no_tags':TAG_NAMES,
    'no_broadcast':{'broadcast_seen'},
    'no_visibility':{'missing','long_gap','broadcast_seen','unknown_players','quality'},
    'motion':{'setpiece','preparation','after_restart','phase','broadcast_seen','unknown_players','quality','contested'},
}


def heuristic(x,names,kind):
    f={n:x[:,i] for i,n in enumerate(names)}
    if kind=='none':return np.zeros(len(x))
    if kind=='tags':return f['preparation']*f['walking']-f['after_restart']
    if kind=='restart':return -f['after_restart']
    if kind=='contact':return -f['contact']*f['ball_motion']*(1-f['setpiece'])
    if kind=='setup':return f['setpiece']*f['walking']
    if kind=='gap_gate':return -f['long_gap']*f['running']*(1-f['setpiece'])
    if kind=='tempo':return f['tempo_drop']*(1-f['ball_motion'])
    raise ValueError('Unknown heuristic: '+kind)


class HalfPossession(AudioPossession):
    def __init__(self,pack,detections):
        super().__init__(pack,detections)
        self.fit_cache={};self.score_cache={};self.audio_cache={}

    def score(self,h,c):
        key=(h,c.get('family','frozen'),c.get('pool','half'),c.get('features','full'),c.get('l2',.1),c.get('window',3),c.get('extra_window',0))
        if key in self.score_cache:return self.score_cache[key].copy()
        x=self.pack[f'x{h}'];family=key[1]
        if family=='frozen':
            score=self.pack[f'score{h}'].copy()
            score=moving_average(score,c.get('extra_window',0))
        elif family=='logistic':
            removed=FEATURE_SETS[c.get('features','full')] or set()
            ix=[i for i,n in enumerate(self.names) if n not in removed]
            pool=c.get('pool','half');l2=c.get('l2',.1)
            fit_key=(h if pool=='half' else -1,tuple(ix),l2)
            if fit_key not in self.fit_cache:
                data=x if pool=='half' else np.vstack([self.pack['x0'],self.pack['x1']])
                chosen,y=physical_seeds(data)
                self.fit_cache[fit_key]=train_logistic(data[chosen][:,ix],y,l2=l2)
            fit=self.fit_cache[fit_key]
            score=((x[:,ix]-fit['mean'])/fit['scale'])@fit['weight']+fit['intercept']
            score=moving_average(score,c.get('window',3))
        elif family=='physics':
            score=evidence(x,dict(family='physics',bias=-.8))
            score=moving_average(score,c.get('window',3))
        else:raise ValueError('Unknown activity family')
        self.score_cache[key]=score
        return score.copy()

    def half(self,h,c):
        a,b=HALVES[h];x=self.pack[f'x{h}'];score=self.score(h,c)
        score+=c.get('bias',0)
        score+=c.get('heuristic_weight',0)*heuristic(x,self.names,c.get('heuristic','none'))
        if c.get('quality_gate'):score*=x[:,self.names.index('quality')]
        audio=dict(mode=c.get('audio_mode','role'),weight=c.get('audio_weight',16),
                   tau=c.get('tau',2),restart_policy=c.get('restart_policy','strict'))
        # No-tag input variants also remove tags from acoustic role context.
        use_tags=c.get('features','full') not in ('no_tags','motion')
        audio_key=(h,audio['tau'],audio['restart_policy'],use_tags)
        if audio_key not in self.audio_cache:
            ctx=x.copy()
            if not use_tags:
                for n in TAG_NAMES:ctx[:,self.names.index(n)]=0
            self.audio_cache[audio_key]=signals(half_events(self.events,a,b),
                (np.arange(a,b,5)+2)/25,ctx,self.pack[f'score{h}'],self.pack[f'state{h}'],self.names,audio)
        signal=self.audio_cache[audio_key]
        if audio['mode']=='role':score+=audio['weight']*signal['role']
        elif audio['mode']=='stop':score+=audio['weight']*signal['recent']
        elif audio['mode']!='none':raise ValueError('Unknown audio mode')
        states=decode(score,c.get('enter',15),c.get('leave',15))
        dead=np.repeat(states,5)[:b-a].astype(bool)
        pred=self.pack['owner'][a:b].copy();pred[dead]=2
        return pred.astype('i1')

    def predict_halves(self,configs):
        from possession.__main__ import submission_labels
        if len(configs)!=len(HALVES):raise ValueError('Exactly one configuration per half required')
        labels=np.full(len(self.pack['owner']),'X',dtype='<U1')
        for h,(a,b) in enumerate(HALVES):labels[a:b]=np.array(list('HADX'))[self.half(h,configs[h])]
        return submission_labels(labels)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,default=Path('outputs/audio_possession_v31/physical_features.npz'))
    p.add_argument('--events',type=Path,default=Path('outputs/whistles_v31/detections.json'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model=HalfPossession(a.pack,json.loads(a.events.read_text()))
    config=json.loads(a.config.read_text())
    labels=model.predict_halves(config['halves'])
    from possession.base import write_prediction
    a.output.parent.mkdir(parents=True,exist_ok=True);write_prediction(a.output,labels)
    print(json.dumps(dict(output=str(a.output),frames=len(labels),uses_references=False)))


if __name__=='__main__':main()
