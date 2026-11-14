"""Geometric observations and temporal decoding."""
from pathlib import Path


import json,sys


import numpy as np


ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


from starter.load_smart_tagging import load as load_tags





from games import asset_paths


from possession.preprocess import HALVES,PITCH,fix_goalkeepers,kalman


from possession.runtime import cache_directory
CACHE=cache_directory();N=199319;STEP=5


from possession.runtime import cache_directory
CACHE=cache_directory();N=199319;STEP=5


from possession.runtime import cache_directory
CACHE=cache_directory();N=199319;STEP=5


def mask_halves(labels):
    out=np.full(N,'X',dtype='<U1')
    for a,b in HALVES:out[a:b]=labels[a:b]
    return out


def tag_features():
    team=np.zeros((N,2));restart=np.zeros(N);inphase=np.zeros(N)
    attack={'PROGRESSION','BUILD UP','ATTACKING TRANSITION','CREATING CHANCES','LONG BALL','BALL IN FINAL THIRD','BALL IN THE BOX'}
    defend={'DEFENDING IN MIDDLE THIRD','DEFENDING IN ATTACKING THIRD','DEFENDING IN DEFENSIVE THIRD','DEFENSIVE TRANSITION'}
    for e in load_tags():
        a=max(0,e['start_frame']);b=min(N,e['end_frame']);name=e['tags'].get('Team');side=0 if name=='Barcelona' else 1 if name in ('Atletic Club','Athletic Club') else None
        code=e['code']
        if side is not None and (code in attack or code in defend):
            if code in defend:side=1-side
            weight=.5 if code in ('BALL IN FINAL THIRD','BALL IN THE BOX') else 1.
            team[a:b,side]=np.maximum(team[a:b,side],weight);inphase[a:b]=1
        if code=='SET PIECES':
            # Windows straddle restart: only the earlier portion is weak dead evidence.
            restart[a:a+(b-a)//2]=1
            if side is not None:team[a:b,side]=np.maximum(team[a:b,side],.7)
    return team,restart,inphase


def inputs(kind):
    CACHE.mkdir(parents=True,exist_ok=True)
    raw_path=CACHE/'tracking_raw.npz'
    if not raw_path.exists():
        from possession.tracking import Source
        source=Source('tactical',correct_goalkeepers=False)
        np.savez_compressed(raw_path,xy=source.xy,ball=source.ball,teams=source.teams,valid=source.valid)
    # Cache provenance covers organiser files; changing input invalidates features.
    paths=asset_paths(source='tactical')
    signature={key:[Path(paths[key]).stat().st_size,Path(paths[key]).stat().st_mtime_ns]
               for key in ('atd_raw','atd_meta','homography','smart_tagging')}
    signature_path=CACHE/'source_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        from possession.tracking import Source
        source=Source('tactical',correct_goalkeepers=False)
        np.savez_compressed(raw_path,xy=source.xy,ball=source.ball,teams=source.teams,valid=source.valid)
        for stale in CACHE.glob('features_*.npz'):stale.unlink()
    signature_path.write_text(json.dumps(signature,sort_keys=True))
    path=CACHE/f'features_{kind}.npz'
    if path.exists():return dict(np.load(path))
    z=np.load(CACHE/'tracking_raw.npz');xy=z['xy'];ball=z['ball'];teams=z['teams']
    if kind!='raw':
        xy,changed=fix_goalkeepers(xy)
        if kind=='keeper_masked':xy[changed]=np.nan
        if kind=='drop_keepers':xy[:,[23,45]]=np.nan
    if kind in ('ball_kf','both_kf','soft_kf','long_kf'):
        ball=kalman(ball,measurement=.8 if kind=='soft_kf' else .3,ball=True,max_gap=2 if kind=='long_kf' else .6)
    if kind in ('both_kf','soft_kf'):xy=kalman(xy,measurement=.4 if kind=='both_kf' else .8,max_gap=.24)
    d=np.linalg.norm((xy-ball[:,None])*PITCH,axis=2);d[~np.isfinite(d)]=1e6
    distance=np.stack([d[:,teams==t].min(1) for t in (0,1)],1)
    close_count=np.stack([(d[:,teams==t]<3).sum(1) for t in (0,1)],1)
    seen=np.isfinite(ball).all(1);original_seen=np.isfinite(z['ball']).all(1)
    speed=np.r_[0,np.linalg.norm(np.diff(ball,axis=0)*PITCH,axis=1)*25];speed=np.nan_to_num(speed,nan=0)
    # Track gaps from the original ball, not from a filter's synthetic predictions.
    gap=np.zeros(N);indices=np.arange(N);last=np.maximum.accumulate(np.where(original_seen,indices,-100000));gap=(indices-last)/25
    nearest_gap=np.minimum(gap,(np.minimum.accumulate(np.where(original_seen,indices,N+100000)[::-1])[::-1]-indices)/25)
    outside=seen&((ball[:,0]<-.005)|(ball[:,0]>1.005)|(ball[:,1]<-.01)|(ball[:,1]>1.01))
    tags,restart,inphase=tag_features()
    f=dict(distance=distance,close_count=close_count,seen=seen,original_seen=original_seen,speed=speed,gap=gap,nearest_gap=nearest_gap,outside=outside,tags=tags,restart=restart,inphase=inphase,valid=z['valid'])
    np.savez_compressed(path,**f);return f


def emissions(f,c):
    d=np.minimum(f['distance'],40);seen=f['seen'];scale=c.get('scale',2.5)
    advantage=np.clip((d[:,1]-d[:,0])/scale,-3,3);strength=np.exp(-np.minimum(d.min(1),40)/c.get('contact',8))
    strength=np.where(seen,strength,0.)
    if c.get('flight',0):strength/=1+(f['speed']/c['flight'])**2
    team=np.stack([advantage,-advantage],1)*strength[:,None]
    # Long absences supply dead evidence; brief occlusions should retain continuity.
    dead=np.where(seen,-c.get('alive',1.8),c.get('missing',.9))
    if c.get('gap',0):
        dead=np.where(seen,-c.get('alive',1.8),np.minimum(f['nearest_gap']/c['gap'],1)*c.get('missing',.9)-.4)
    dead+=f['outside']*c.get('outside',2.)
    dead+=f['restart']*c.get('restart',0)
    dead-=(f['speed']>2)*seen*c.get('moving',0)
    weight=c.get('tags',0);team+=weight*(f['tags']-.5*f['tags'][:,::-1])
    if c.get('phase_live',0):dead-=f['inphase']*c['phase_live']
    if c.get('crowd',0):team+=np.clip(f['close_count']-f['close_count'][:,::-1],-2,2)*c['crowd']
    if c.get('invalid',False):team[~f['valid']]=0
    return np.c_[team,dead]


def viterbi(log_emission,switch,dead_switch,min_dwell=1,transition=None):
    """3-state HMM with fixed symmetric transition priors and Viterbi decoding.

    Optional state expansion enforces a minimum dwell (duration-constrained HMM).
    Emission potentials are normalized per observation; normalizing adds a common
    term and does not alter the maximum-probability path.
    """
    m=min_dwell;k=3*m;state_class=np.repeat(np.arange(3),m);trans=np.full((k,k),-1e9)
    for label in range(3):
        base=label*m
        for age in range(m-1):trans[base+age,base+age+1]=0
        trans[base+m-1,base+m-1]=0
        for other in range(3):
            if other!=label:trans[base+m-1,other*m]=-(dead_switch if label==2 or other==2 else switch)
    # Normalize each row to obtain a valid transition probability matrix.
    high=trans.max(1);trans-= (high+np.log(np.exp(trans-high[:,None]).sum(1)))[:,None]
    if transition is not None:trans=np.log(transition)
    back=np.empty((len(log_emission),k),dtype=np.int16);score=np.full(k,-1e9);score[np.arange(3)*m]=0
    for t,e in enumerate(log_emission):
        paths=score[:,None]+trans;idx=paths.argmax(0);back[t]=idx;score=paths[idx,np.arange(k)]+e[state_class];score-=score.max()
    out=np.empty(len(log_emission),dtype=np.int8);j=score.argmax()
    for t in range(len(out)-1,-1,-1):out[t]=state_class[j];j=back[t,j]
    return out


def write_prediction(path,labels):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        f.write('frame,label\n');f.writelines(f'{i},{x}\n' for i,x in enumerate(labels))


def predict(config):
    from possession.ownership import strategy_predict
    return strategy_predict(config)
