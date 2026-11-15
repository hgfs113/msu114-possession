"""Receive/hold/release versus restart preparation, using organiser evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from possession.preprocess import HALVES
from possession.whistle.detect import sha
from possession.whistle.half_model import HalfPossession,heuristic
from possession.whistle.activity_v31 import half_events,signals
from possession.whistle.fusion import decode
from possession.goalkeeper.model import COMMON

PHASE_NAMES=('LIVE','DEAD','RECEIVE_H','HOLD_H','RELEASE_H','SETUP_H','RECEIVE_A','HOLD_A','RELEASE_A','SETUP_A')


def recent_max(x,width):
    out=x.copy()
    for lag in range(1,width+1):out[lag:]=np.maximum(out[lag:],x[:-lag])
    return out


def phase_evidence(f,h,base_live,audio,c):
    d=f[f'distance{h}'];seen=f[f'seen{h}']>.5;ks=f[f'keeper_seen{h}']>.5;box=f[f'keeper_box{h}']>.5
    ballbox=f[f'ball_box{h}']>.5;speed=f[f'ball_speed{h}'];kv=f[f'keeper_speed{h}'];rad=f[f'radial{h}']
    relv=f[f'relative_speed{h}'];other=f[f'other{h}'];n=len(d)
    near=seen[:,None]&ks&box&ballbox&(d<3)&(other[:,None]>d+.5)
    still=seen&(speed<1);goal_still=still[:,None]&(f[f'goal_area{h}']>.5)&box&(d<8)
    outside=recent_max(f[f'outside{h}'],50)
    stop=recent_max(np.clip(audio['role'],0,1),15)
    goal_tag=f[f'goal_tag{h}']
    setup=np.maximum(goal_tag[:,None]*goal_still,np.maximum(.9*outside[:,None]*goal_still,.7*stop[:,None]*goal_still))
    arrivals=np.zeros_like(d);source=np.full(d.shape,'',dtype='<U12')
    style=c.get('anchor','mixed')
    for team in (0,1):
        for i in np.flatnonzero(near[:,team]):
            start=max(0,i-10);prior=(seen[start:i]&(d[start:i,team]>5)&(speed[start:i]>3))
            approaching=(rad[start:i,team]<-1.5)&prior
            future=min(n,i+7)
            stopped=(speed[i:future]<3)&seen[i:future]
            hidden=(~seen[i:future]).sum()>=3
            if approaching.any() and (stopped.any() or hidden):arrivals[i,team]=1;source[i,team]='observed'
        if style in ('mixed','carry'):
            for i in range(n-6):
                if not(seen[i] and ks[i,team] and box[i,team] and ballbox[i,team]):continue
                if not(3<d[i,team]<9 and 0<f[f'tca{h}'][i,team]<1.2 and f[f'miss_distance{h}'][i,team]<2.5 and rad[i,team]<-2):continue
                if seen[i+1:i+4].any():continue
                j=min(n-1,i+max(1,round(float(f[f'tca{h}'][i,team])*5)))
                if ks[i:j+1,team].all() and box[i:j+1,team].all():
                    if arrivals[j,team]<.75:arrivals[j,team]=.75;source[j,team]='occluded'
        if style=='carry':
            moving=near[:,team]&(kv[:,team]>.25)&(relv[:,team]<2)
            for i in range(5,n):
                if not moving[i] or moving[i-5:i+1].sum()<4:continue
                displacement=np.linalg.norm(f[f'keeper_xy{h}'][i,team]-f[f'keeper_xy{h}'][i-5,team])
                if np.isfinite(displacement) and displacement>.6 and arrivals[i,team]<.85:
                    arrivals[i,team]=.85;source[i,team]='co_motion'
    receiving=np.zeros_like(d);holding=np.zeros_like(d);releasing=np.zeros_like(d);episodes=[]
    new_stop=(audio['role']>.25)&np.r_[True,audio['role'][:-1]<=.25]
    gate=c.get('gate','context');require_release=c.get('require_release',True)
    for team in (0,1):
        busy=-1
        for i in np.flatnonzero(arrivals[:,team]>0):
            if i<=busy:continue
            before=base_live[max(0,i-15):max(0,i-2)];prior_live=float(before.mean()) if len(before) else 0.
            context=max(float(setup[max(0,i-5):i+1,team].max()),float(outside[i])*.9,float(stop[i])*.7)
            if gate!='none' and context>.65:continue
            if gate=='strict' and (prior_live<.6 or goal_tag[i]>.5 or stop[i]>.2):continue
            limit=min(n,i+225);end=limit;release=None;reason='limit';bad_keeper=0;far=0;standard=0
            for j in range(i+1,limit):
                bad_keeper=bad_keeper+1 if not(ks[j,team] and box[j,team]) else 0
                far=far+1 if seen[j] and d[j,team]>12 else 0
                standard=standard+1 if setup[j,team]>.65 else 0
                depart=seen[j] and 3<d[j,team]<35 and speed[j]>4 and rad[j,team]>3
                if depart and j>i+2:
                    release=j;end=j;reason='release';break
                if bad_keeper>=10:end=j-9;reason='keeper_lost';break
                if new_stop[j] and gate!='none':end=j;reason='stop_whistle';break
                if standard>=5 and gate!='none':end=j-4;reason='restart_setup';break
                if far>=2:end=j-1;reason='observed_elsewhere';break
            if end<=i+2:continue
            busy=end
            accepted=release is not None or not require_release
            episodes.append(dict(half=h,team=team,start=int(i),end=int(end),release=None if release is None else int(release),
                source=str(source[i,team]),prior_live=prior_live,standard_context=context,end_reason=reason,accepted=accepted))
            if not accepted:continue
            anchor=float(arrivals[i,team]);cut=min(end,i+3)
            receiving[i:cut,team]=anchor
            support=np.where(near[cut:end,team],1.,.8)*ks[cut:end,team]*box[cut:end,team]
            if gate!='none':support*=1-.8*setup[cut:end,team]
            holding[cut:end,team]=anchor*support
            if release is not None:releasing[release:min(n,release+4),team]=anchor
    return dict(receive=receiving,hold=holding,release=releasing,setup=setup,episodes=episodes,arrivals=arrivals)


def decode_phases(score,p,weight=8):
    """Ten-state constrained max-sum decoder. Additional states need observations."""
    n=len(score);trans=np.full((10,10),-1e4);np.fill_diagonal(trans,0)
    trans[0,1]=trans[1,0]=-15
    emission=np.full((n,10),-1e4);emission[:,0]=0;emission[:,1]=score
    for t in (0,1):
        r,k,l,s=2+4*t,3+4*t,4+4*t,5+4*t
        trans[0,r]=-4;trans[1,r]=-19;trans[r,k]=-.2
        trans[r,0]=trans[k,0]=-2;trans[r,1]=trans[k,1]=-15
        trans[k,l]=-.5;trans[r,l]=-.5;trans[l,0]=-.5;trans[l,1]=-15
        trans[1,s]=trans[s,1]=-.5;trans[0,s]=-15;trans[s,0]=-15;trans[s,l]=-15
        for state,key in ((r,'receive'),(k,'hold'),(l,'release')):
            support=p[key][:,t];emission[:,state]=np.where(support>.05,weight*support-2,-1e4)
        support=p['setup'][:,t];emission[:,s]=np.where(support>.25,score+weight*.5*support-2,-1e4)
    values=np.full(10,-1e4);values[:2]=0;back=np.empty((n,10),dtype='i1')
    for i,e in enumerate(emission):
        paths=values[:,None]+trans;prev=paths.argmax(0);back[i]=prev
        values=paths[prev,np.arange(10)]+e;values-=values.max()
    state=int(values.argmax());path=np.empty(n,dtype='i1')
    for i in range(n-1,-1,-1):path[i]=state;state=int(back[i,state])
    return path


class KeeperPhaseModel:
    def __init__(self,features='outputs/keeper_phase/features.npz', physical_pack='outputs/audio_possession_v31/physical_features.npz', detections='outputs/whistles_v31/detections.json'):
        parent=HalfPossession(physical_pack,json.loads(Path(detections).read_text()))
        self.pack=parent.pack;self.base=[];self.cache={}
        path=Path(features);meta=json.loads(path.with_suffix('.json').read_text())
        if sha(path)!=meta['pack_sha256']:raise ValueError('Keeper phase features changed')
        self.features=dict(np.load(path,allow_pickle=False))
        for h,(a,b) in enumerate(HALVES):
            x=parent.pack[f'x{h}']
            audio=signals(half_events(parent.events,a,b),(np.arange(a,b,5)+2)/25,x,parent.pack[f'score{h}'],parent.pack[f'state{h}'],parent.names,COMMON)
            score=parent.pack[f'score{h}']+2*heuristic(x,parent.names,'tags')+16*audio['role']
            self.base.append(dict(score=score,dead=decode(score,15,15),audio=audio))

    def predict(self,c):
        from possession.__main__ import submission_labels
        labels=np.full(len(self.pack['owner']),'X',dtype='<U1');all_phases=[];details=[]
        mode=c.get('mode','control')
        key=json.dumps({k:v for k,v in c.items() if k not in ('name','mode','weight')},sort_keys=True)
        if key not in self.cache:self.cache[key]=[phase_evidence(self.features,h,self.base[h]['dead']==0,self.base[h]['audio'],c) for h in (0,1)]
        for h,(a,b) in enumerate(HALVES):
            p=self.cache[key][h];base=self.base[h];score=base['score'].copy();owner=self.pack['owner'][a:b:5].copy();weight=c.get('weight',8)
            support=np.maximum(p['receive'],p['hold']);path=base['dead'].copy()
            if mode=='phase':
                path=decode_phases(score,p,weight);dead=np.isin(path,[1,5,9]);owner[dead]=2
                owner[np.isin(path,[2,3])]=0;owner[np.isin(path,[6,7])]=1
            elif mode in ('control','soft','override'):
                if mode=='soft':score-=weight*support.max(1);score+=weight*.5*p['setup'].max(1)
                dead=decode(score,15,15);owner[dead==1]=2;path=dead.copy()
                if mode in ('soft','override'):
                    take=support.max(1)>.5
                    if mode=='soft':take&=dead==0
                    team=support.argmax(1);owner[take]=team[take]
                    for t in (0,1):
                        active=take&(team==t);path[active]=3+4*t
                        path[active&(p['receive'][:,t]>.5)]=2+4*t
            else:raise ValueError(mode)
            labels[a:b]=np.repeat(np.array(list('HADX'))[owner],5)[:b-a];all_phases.append(path);details.extend(p['episodes'])
        return submission_labels(labels),all_phases,details


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();labels,phases,details=KeeperPhaseModel().predict(json.loads(a.config.read_text()))
    from possession.base import write_prediction
    a.output.parent.mkdir(parents=True,exist_ok=True);write_prediction(a.output,labels)
    print(json.dumps(dict(output=str(a.output),episodes=len(details),uses_references=False)))


if __name__=='__main__':main()
