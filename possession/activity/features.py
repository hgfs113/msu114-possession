from functools import lru_cache


import hashlib,json


import numpy as np


from possession.base import N,CACHE,inputs


from possession.preprocess import HALVES,PITCH,fix_goalkeepers


from possession.kinematics import kinematics,sustained_contact


from possession.ownership import smooth_signal


from possession.motion import motion_features


from possession.context import stoppage_features,homography_features


from possession.cameras import aligned_broadcast,signature


NAMES=('missing','long_gap','still','ball_motion','walking','running','unknown_players',
       'setpiece','preparation','after_restart','outside','contact','contested','phase',
       'broadcast_seen','pressure','tempo_drop','quality')


def reduce_blocks(array,halves=HALVES,step=5):
    return [np.asarray([array[i:min(i+step,b)].mean(axis=0) for i in range(a,b,step)]) for a,b in halves]


@lru_cache(maxsize=1)
def features():
    base=inputs('fixed');key=hashlib.sha256((json.dumps(signature(),sort_keys=True)+'dead-features-v1').encode()).hexdigest()[:12]
    path=CACHE/f'dead_features_{key}.npz'
    if path.exists():return dict(np.load(path))
    k=kinematics('adaptive_rts');motion=motion_features();stop=stoppage_features(4,1.8);h=homography_features();bc=aligned_broadcast()
    observed=k['original_seen'];players=motion['observed_count']>=6
    running=np.divide(motion['moving'],motion['observed_count'],out=np.zeros(N),where=players)
    raw=np.load(CACHE/'tracking_raw.npz');xy,_=fix_goalkeepers(raw['xy']);xy=xy*PITCH;ball=raw['ball']*PITCH
    velocity=np.full_like(xy,np.nan)
    for a,b in HALVES:velocity[a+5:b-5]=(xy[a+10:b]-xy[a:b-10])/.4
    speed=np.linalg.norm(velocity,axis=2);delta=ball[:,None]-xy;distance=np.linalg.norm(delta,axis=2)
    radial=np.divide(np.sum(velocity*delta,axis=2),distance,out=np.zeros_like(distance),where=distance>.1)
    approaching=np.isfinite(speed)&(speed<12)&(distance<12)&(radial>1)&observed[:,None]
    pressure=np.clip(approaching.sum(1)/3,0,1)
    contacts=sustained_contact(k,dict(radius=1,relative_speed=3,contact_frames=3)).any(1)
    values=dict(missing=(~observed).astype(float),long_gap=np.clip(k['nearest_gap']/3,0,1)*(~observed),
                still=observed&(k['speed']<.5),ball_motion=observed&(k['speed']>3),
                walking=players&(motion['median_speed']<1.8),running=running,unknown_players=~players,
                setpiece=stop['setup'],preparation=stop['preparation'],after_restart=stop['after_restart'],
                outside=k['outside'],contact=contacts,contested=observed&(k['distance'].max(1)<5),
                phase=base['inphase'],broadcast_seen=bc['seen'],pressure=pressure,
                tempo_drop=np.clip(smooth_signal(running,4)-running,0,1),quality=1/(1+h['error']**2))
    x=np.column_stack([smooth_signal(values[name].astype(float),1) for name in NAMES]).astype('f4')
    active=np.zeros(N,bool)
    for a,b in HALVES:active[a:b]=True
    x[~active]=0
    assert np.isfinite(x).all()
    out=dict(x=x,active=active);np.savez_compressed(path,**out);return out

