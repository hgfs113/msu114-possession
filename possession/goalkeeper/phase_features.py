"""Observed approach, carry, release and restart context. No reference inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from possession.preprocess import HALVES,PITCH,fix_goalkeepers
from possession.whistle.detect import sha


def sample_ball(ball,idx,a,b):
    result=np.full((len(idx),2),np.nan);frame=idx.copy()
    for offset in (0,-1,1,-2,2):
        at=np.clip(idx+offset,a,b-1);take=~np.isfinite(result).all(1)&np.isfinite(ball[at]).all(1)
        result[take]=ball[at[take]];frame[take]=at[take]
    return result,frame


def measured_velocity(position,times,max_speed):
    velocity=np.full_like(position,np.nan,dtype=float)
    if len(position)<2:return velocity
    dt=np.diff(times);delta=np.diff(position,axis=0)
    v=delta/np.maximum(dt,1e-8).reshape((-1,)+(1,)*(delta.ndim-1))
    valid=np.isfinite(v).all(-1)&(np.linalg.norm(v,axis=-1)<max_speed)
    velocity[1:]=np.where(valid[...,None],v,np.nan)
    return velocity


def prepare(raw_path,output,data_dir):
    from starter.load_smart_tagging import load,pattern_path
    raw=dict(np.load(raw_path,allow_pickle=False));xy,corrected=fix_goalkeepers(raw['xy'])
    xy=xy*PITCH;ball=raw['ball']*PITCH;tags=load(data_dir=data_dir)
    windows=[e for e in tags if e['code']=='SET PIECES' and e['tags'].get('Type')=='Goal Kick']
    result={}
    for h,(a,b) in enumerate(HALVES):
        idx=np.minimum(np.arange(a,b,5)+2,b-1);bp,bf=sample_ball(ball,idx,a,b)
        kp=xy[idx][:,[23,45]];seen=np.isfinite(bp).all(1);ks=np.isfinite(kp).all(2)
        vb=measured_velocity(bp,bf/25,45);vk=measured_velocity(kp,idx/25,12)
        dist=np.linalg.norm(bp[:,None]-kp,axis=2)
        field=np.delete(xy[idx],[23,45],axis=1)
        other=np.min(np.nan_to_num(np.linalg.norm(field-bp[:,None],axis=2),nan=1000),axis=1)
        rel=bp[:,None]-kp
        radial=np.sum((vb[:,None]-vk)*rel,axis=2)/np.maximum(dist,.1)
        rel_speed=np.linalg.norm(vb[:,None]-vk,axis=2)
        # Time and perpendicular distance to the keeper on the current measured ray.
        vv=vb[:,None]-vk;v2=np.sum(vv*vv,2)
        tca=-np.sum(rel*vv,2)/np.maximum(v2,1e-6)
        miss=np.linalg.norm(rel+np.maximum(tca,0)[:,:,None]*vv,axis=2)
        own_kx=np.column_stack([kp[:,0,0] if h==0 else PITCH[0]-kp[:,0,0],
            kp[:,1,0] if h==1 else PITCH[0]-kp[:,1,0]])
        own_bx=np.column_stack([bp[:,0] if h==0 else PITCH[0]-bp[:,0],bp[:,0] if h==1 else PITCH[0]-bp[:,0]])
        kbox=ks&(own_kx>=-1)&(own_kx<21)&(abs(kp[:,:,1]-34)<23)
        bbox=seen[:,None]&(own_bx>=-.5)&(own_bx<23)&(abs(bp[:,1,None]-34)<24)
        goal_area=seen[:,None]&(own_bx>=-.5)&(own_bx<7)&(abs(bp[:,1,None]-34)<12)
        outside=seen&((bp[:,0]<-.5)|(bp[:,0]>PITCH[0]+.5)|(bp[:,1]<-.5)|(bp[:,1]>PITCH[1]+.5))
        goal_tag=np.zeros(len(idx))
        for e in windows:goal_tag[(idx>=e['start_frame'])&(idx<e['end_frame'])]=1
        values=dict(distance=np.nan_to_num(dist,nan=1000),other=other,seen=seen,keeper_seen=ks,keeper_box=kbox,ball_box=bbox,
            ball_speed=np.linalg.norm(vb,axis=1),keeper_speed=np.linalg.norm(vk,axis=2),radial=radial,
            relative_speed=rel_speed,tca=tca,miss_distance=miss,goal_area=goal_area,outside=outside,goal_tag=goal_tag,
            keeper_xy=kp,ball_xy=bp,corrected=corrected[idx][:,[23,45]])
        for key,value in values.items():result[f'{key}{h}']=np.asarray(value,dtype='f4')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)
    output.with_suffix('.json').write_text(json.dumps(dict(pack_sha256=sha(output),raw_sha256=sha(raw_path),
        tag_sha256=sha(pattern_path(data_dir=data_dir)),goal_kick_windows=windows,uses_references=False,
        note='5 Hz measured ball samples, no interpolation; unknown velocity stays missing; corrected keeper-side hypothesis'),indent=2)+'\n')
    print(json.dumps(dict(goal_kicks=len(windows),tag_durations_seconds=sorted(set(e['end_s']-e['start_s'] for e in windows)))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('.cache/whistle/e053260cf078156a1805/tracking_raw.npz'))
    p.add_argument('--output',type=Path,default=Path('outputs/keeper_phase/features.npz'))
    p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args();prepare(a.raw,a.output,a.data_dir)


if __name__=='__main__':main()
