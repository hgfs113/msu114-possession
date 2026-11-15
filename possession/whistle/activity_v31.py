"""Combine fixed acoustic and physical evidence separately in each half."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .fusion import decode, context_score
from ..preprocess import HALVES


def half_events(events,a,b):
    """Peak must be within an active half; clip acoustic support to that half."""
    return [dict(e,start_seconds=max(a/25,e['start_seconds']),end_seconds=min(b/25,e['end_seconds']))
            for e in events if a/25<=e['peak_seconds']<b/25]


def prepare_pack(destination):
    from possession.activity.features import features,reduce_blocks,NAMES
    from possession.activity.model import evidence,moving_average,freeze_ownership,binary_decode
    from possession.runtime import cache_directory
    from possession.cameras import signature
    from .detect import sha
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(__file__).parents[1].joinpath('activity/interval_config.json').read_text())
    blocks=reduce_blocks(features()['x']);_,owner=freeze_ownership()
    packed=dict(owner=owner,names=np.array(NAMES),config=np.array(json.dumps(config)))
    for h,x in enumerate(blocks):
        score=moving_average(evidence(x,config),config.get('window',3))
        packed[f'x{h}']=x;packed[f'score{h}']=score
        packed[f'state{h}']=binary_decode(score,config['enter'],config['leave'])
    np.savez_compressed(destination,**packed)
    destination.with_suffix('.json').write_text(json.dumps(dict(source='organiser tracking/tags/homography',
        assets=signature(),cache=str(cache_directory()),pack_sha256=sha(destination),
        algorithm_config=config,uses_references=False),indent=2)+'\n')


def neighbourhood(x,t,lo,hi):
    a=max(0,t+round(lo*5));b=min(len(x),t+round(hi*5))
    return x[a:b].mean(0) if b>a else x[min(max(t,0),len(x)-1)]


def signals(events,centres,x,score,state,names,config):
    near=np.zeros(len(centres));recent=near.copy();signed=near.copy();role=near.copy()
    ctx=context_score(x,names);roles=[]
    tau=config.get('tau',2);shift=config.get('offset',0)
    for event in events:
        origin=event['start_seconds']+shift
        delta=centres-origin
        pulse=np.exp(-.5*(delta/.65)**2)*(abs(delta)<2)
        tail=np.exp(-np.maximum(delta,0)/tau)*((delta>=0)&(delta<4*tau))
        near=np.maximum(near,pulse);recent=np.maximum(recent,tail)
        i=int(np.clip(np.searchsorted(centres,origin),0,len(centres)-1))
        before=neighbourhood(state,i,-3,-.4)
        future=neighbourhood(x,i,.4,4)
        motion=future[names.index('ball_motion')];restart=future[names.index('after_restart')]
        stop_context=float(neighbourhood(ctx,i,.2,4))
        # A signal after established DEAD may authorise a restart; movement must support it.
        restart_strength=float(np.clip(before,0,1)*np.clip(1.5*motion+restart,0,1))
        policy=config.get('restart_policy','permissive')
        if policy=='none':restart_strength=0.
        elif policy=='tag':restart_strength=float(before*np.clip(restart,0,1))
        elif policy=='strict':
            clear_motion=(motion>.6 and future[names.index('contact')]>.15 and
                          future[names.index('running')]>.25 and future[names.index('setpiece')]<.2)
            restart_strength=float(before*max(np.clip(restart,0,1),float(clear_motion)))
        elif policy=='contact':
            restart_strength=float(before*np.clip(motion*future[names.index('contact')]*4+restart,0,1))
        elif policy!='permissive':raise ValueError('Unknown restart policy')
        stop_strength=float((1-before)*(.35+.65*np.clip(stop_context+.3,0,1)))
        value=stop_strength-restart_strength
        confidence=float(np.clip((event.get('score_db',28)-14)/14,.4,1)) if config.get('acoustic_weight') else 1.
        role+=tail*value*confidence
        signed+=tail*stop_context
        roles.append(dict(id=event['id'],start_seconds=event['start_seconds'],
                          prior_dead=float(before),post_motion=float(motion),
                          role_score=value,context_score=stop_context))
    return dict(near=near,recent=recent,signed=np.clip(signed,-1,1),role=np.clip(role,-1,1),context=ctx,events=roles)


def snap_boundaries(states,events,centres,radius):
    """Align an existing state boundary without adding new state changes."""
    original=states.copy();result=states.copy()
    edges=np.flatnonzero(np.diff(original)!=0)+1
    for edge in edges:
        nearby=[e for e in events if abs(e['start_seconds']-centres[edge])<=radius]
        if not nearby:continue
        event=min(nearby,key=lambda e:abs(e['start_seconds']-centres[edge]))
        dest=int(np.clip(np.searchsorted(centres,event['start_seconds']),1,len(states)-1))
        # Do not cross another established boundary or collapse neighbouring spells.
        if any(min(edge,dest)<other<=max(edge,dest) for other in edges if other!=edge):continue
        if dest<edge:result[dest:edge]=original[edge]
        elif dest>edge:result[edge:dest]=original[edge-1]
    return result


class AudioPossession:
    def __init__(self,pack,detections):
        from .detect import sha
        pack=Path(pack);meta=json.loads(pack.with_suffix('.json').read_text())
        if meta['pack_sha256']!=sha(pack):raise ValueError('Physical feature pack changed')
        self.pack=dict(np.load(pack,allow_pickle=False))
        if detections.get('recording_source')!='tactical':raise ValueError('Expected tactical audio clock')
        self.events=detections['events'];self.names=self.pack['names'].tolist()
        self.config=json.loads(str(self.pack['config']))

    def predict(self,c):
        from possession.__main__ import submission_labels
        owner=self.pack['owner'];activity=np.full(len(owner),'X',dtype='<U1');diagnostics=[]
        for h,(a,b) in enumerate(HALVES):
            x=self.pack[f'x{h}'];score=self.pack[f'score{h}'].copy();base=self.pack[f'state{h}']
            events=half_events(self.events,a,b)
            centres=(np.arange(a,b,5)+2)/25
            audio=signals(events,centres,x,score,base,self.names,c)
            mode=c['mode'];weight=c.get('weight',0)
            if mode=='stop':score+=weight*audio['recent']
            elif mode=='signed':score+=weight*audio['recent']*audio['context']
            elif mode=='event_context':score+=weight*audio['signed']
            elif mode=='role':score+=weight*audio['role']
            elif mode not in ('control','boundary','snap'):raise ValueError('Unknown fusion mode')
            states=decode(score,self.config['enter'],self.config['leave'],boundary=audio['near'],
                          strength=c.get('strength',0),context=audio['context'] if c.get('context_gate') else None)
            if mode=='snap':states=snap_boundaries(states,events,centres,c['radius'])
            activity[a:b]=np.repeat(np.array(['L','D'])[states],5)[:b-a]
            diagnostics.extend(audio['events'])
        labels=np.array(list('HADX'))[owner];labels[activity=='D']='D';labels[activity=='X']='X'
        return submission_labels(labels),activity,diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,required=True);p.add_argument('--events',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--data-dir',type=Path)
    p.add_argument('--cache-dir',type=Path,default=Path('.cache/whistle'))
    a=p.parse_args()
    if a.data_dir:os.environ['MTAH_DATA_DIR']=str(a.data_dir.resolve())
    os.environ['POSSESSION_CACHE_DIR']=str(a.cache_dir.resolve())
    if a.prepare:prepare_pack(a.pack);return
    if not(a.events and a.config and a.output):p.error('--events, --config and --output are required')
    from possession.base import write_prediction
    labels,_,_=AudioPossession(a.pack,json.loads(a.events.read_text())).predict(json.loads(a.config.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True);write_prediction(a.output,labels)
    print(json.dumps(dict(frames=len(labels),output=str(a.output),uses_references=False)))


if __name__=='__main__':main()
