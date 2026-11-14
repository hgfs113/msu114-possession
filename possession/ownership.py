import numpy as np


from functools import lru_cache


from possession.base import N,HALVES,viterbi


from possession.kinematics import contact_predict,decode_general,kinematics,sustained_contact


from starter.load_smart_tagging import load as load_tags


def smooth_signal(x,seconds,halves=HALVES):
    """Two box passes give triangular smoothing without leaking across halftime."""
    out=np.zeros_like(x,dtype=float);w=max(0,round(seconds*25/2))
    for a,b in halves:
        y=x[a:b].astype(float)
        for _ in range(2):
            ix=np.arange(len(y));lo=np.maximum(0,ix-w);hi=np.minimum(len(y),ix+w+1);cs=np.concatenate([np.zeros_like(y[:1]),y.cumsum(0)],axis=0)
            den=(hi-lo).reshape((-1,)+(1,)*(y.ndim-1));y=(cs[hi]-cs[lo])/den
        out[a:b]=y
    return out


@lru_cache(maxsize=16)
def tag_signals(seconds=2.,mode='all',calibrated=False):
    attack={'PROGRESSION','BUILD UP','ATTACKING TRANSITION','CREATING CHANCES','LONG BALL','BALL IN FINAL THIRD','BALL IN THE BOX'}
    defend={'DEFENDING IN MIDDLE THIRD','DEFENDING IN ATTACKING THIRD','DEFENDING IN DEFENSIVE THIRD','DEFENSIVE TRANSITION'}
    codes=sorted(attack|defend);signals={code:np.zeros(N) for code in codes};stop=np.zeros(N)
    for e in load_tags():
        a=max(0,e['start_frame']);b=min(N,e['end_frame']);team=e['tags'].get('Team');side=1 if team=='Barcelona' else -1 if team in ('Atletic Club','Athletic Club') else 0;code=e['code']
        if code in signals:
            if code in defend:side=-side
            if mode=='attack' and code not in attack:continue
            signals[code][a:b]=side
        if code=='SET PIECES':stop[a:a+(b-a)//2]=1
    weights={code:1. for code in codes}
    if calibrated:
        f=kinematics('adaptive_rts');contact=sustained_contact(f,dict(radius=1,relative_speed=3,contact_frames=3));observed=contact[:,0].astype(float)-contact[:,1]
        # Shrunk tag reliability estimated from automatic physical contacts.
        for code,s in signals.items():
            sample=(s!=0)&(observed!=0);n=int(sample.sum());right=int((s[sample]==observed[sample]).sum())
            p=(right+250)/(n+500);weights[code]=max(0.,2*p-1)
    numerator=np.zeros(N);denominator=np.zeros(N)
    for code,s in signals.items():numerator+=s*weights[code];denominator+=(s!=0)*weights[code]
    # No double confidence from correlated attack/defence tags.
    signed=np.divide(numerator,denominator,out=np.zeros(N),where=denominator>0)
    return smooth_signal(signed,seconds),smooth_signal(stop,seconds),weights


def reduce_emissions(e,step=5,window=5):
    blocks=[]
    for a,b in HALVES:
        red=np.array([e[i:min(i+step,b)].mean(0) for i in range(a,b,step)])
        if window:
            ix=np.arange(len(red));lo=np.maximum(0,ix-window);hi=np.minimum(len(red),ix+window+1);cs=np.vstack([np.zeros(3),np.cumsum(red,axis=0)]);red=(cs[hi]-cs[lo])/(hi-lo)[:,None]
        blocks.append(red)
    return blocks


def strategy_emissions(c):
    e,contact,f=contact_predict(c,return_emissions=True)
    tags,stops,weights=tag_signals(c.get('tag_smooth',2),c.get('tag_mode','all'),c.get('tag_calibrated',False))
    gate=np.ones(N)
    if c.get('tag_gate')=='missing':gate=(~f['original_seen']).astype(float)
    if c.get('tag_gate')=='ambiguous':
        near=smooth_signal(contact.any(1).astype(float),1.)
        gate=np.clip(1-near*4,0,1)
    signed=tags*c.get('tag_weight',0)*gate;e[:,0]+=signed;e[:,1]-=signed
    e[:,2]+=stops*c.get('tag_dead',0)
    if c.get('restart_gate',False):e[:,2]-=stops*c.get('tag_dead',0)*f['original_seen']
    if c.get('team_motion',0):
        # Team movement proxy is independent of unstable player identities.
        from possession.motion import motion_features
        motion=motion_features();e[:,2]-=np.clip(motion['median_speed']-1,0,2)*c['team_motion']*(~f['original_seen'])
    extra=dict(tags=tags,stops=stops,weights=weights)
    if c.get('stoppage_mode') or c.get('homography_mode') or c.get('broadcast_weight'):
        from possession.context import augment_emissions
        e,diagnostic=augment_emissions(e,contact,f,c);extra.update(diagnostic)
    return e,contact,f,extra


def strategy_predict(c):
    e, _, _, _ = strategy_emissions(c)
    out = np.full(N, 'X', dtype='<U1')
    for (a, b), red in zip(HALVES, reduce_emissions(e, 5, c.get('window', 5))):
        states = viterbi(red, c.get('switch', 5), c.get('dead_switch', 8), c.get('dwell', 1))
        out[a:b] = np.repeat(np.array(['H', 'A', 'D'])[states], 5)[:b-a]
    return out
