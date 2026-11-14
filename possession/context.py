from functools import lru_cache


import numpy as np


from possession.base import N, CACHE, inputs


from possession.preprocess import HALVES, PITCH


from possession.kinematics import kinematics


from possession.ownership import smooth_signal


from possession.motion import motion_features


from starter.load_smart_tagging import load as load_tags


from starter.load_homography import load as load_homography


def project_batch(h, points):
    homogeneous=np.concatenate([points,np.ones(points.shape[:-1]+(1,))],axis=-1)
    pixels=np.einsum('nij,n...j->n...i',h,homogeneous)
    denominator=pixels[...,2:3]
    return np.divide(pixels[...,:2],denominator,out=np.full_like(pixels[...,:2],np.nan),where=abs(denominator)>1e-9)


@lru_cache(maxsize=1)
def homography_features():
    inputs('fixed');raw=np.load(CACHE/'tracking_raw.npz')
    h,error,valid=load_homography(source='tactical')
    pixel=project_batch(h,raw['ball'])
    # Actual local tactical MP4 metadata: 1920x1080. No pixel data read here.
    in_image=np.isfinite(pixel).all(1)&(pixel[:,0]>=0)&(pixel[:,0]<1920)&(pixel[:,1]>=0)&(pixel[:,1]<1080)
    grid=np.array([[x,y] for x in np.linspace(0,1,9) for y in np.linspace(0,1,7)])
    corners=project_batch(h,np.broadcast_to(grid,(N,len(grid),2)))
    field_fraction=((corners[...,0]>=0)&(corners[...,0]<1920)&(corners[...,1]>=0)&(corners[...,1]<1080)).mean(1)
    return dict(error=error,valid=valid,in_image=in_image,field_fraction=field_fraction)


@lru_cache(maxsize=12)
def stoppage_features(speed_threshold=2.5,walking_threshold=1.5):
    f=kinematics('adaptive_rts');motion=motion_features()
    # Unknown player velocity must not masquerade as walking/stationary players.
    slow=smooth_signal(((motion['median_speed']<walking_threshold)&(motion['observed_count']>=6)).astype(float),1)
    speed=f['speed'];observed=f['original_seen']
    still=smooth_signal((observed&(speed<.5)).astype(float),1)
    # Restart = observed sustained movement following a stationary / unobserved spell.
    moving=smooth_signal((observed&(speed>speed_threshold)).astype(float),.4)
    preparation=np.zeros(N);after_restart=np.zeros(N);setup=np.zeros(N);events=[]
    for event in load_tags():
        if event['code']!='SET PIECES':continue
        for lo,hi in HALVES:
            a=max(lo,event['start_frame']);b=min(hi,event['end_frame'])
            if b<=a:continue
            setup[a:b]=1
            # Find a movement onset inside the window, not an arbitrary midpoint.
            search=a+min(25,(b-a)//4)
            candidates=[]
            for t in range(search,b-10,5):
                past=max(a,t-50)
                quiet=((speed[past:t]<1.)|(~observed[past:t])).mean()
                if moving[t]>.65 and quiet>.65:
                    support=quiet+.4*slow[past:t].mean()+.2*still[past:t].mean()
                    candidates.append((support,t))
            if candidates:
                _,restart=max(candidates)
                # This boundary is inferred from local movement observations.
                preparation[a:restart]=np.maximum(preparation[a:restart],.4+.6*slow[a:restart])
                after_restart[restart:min(b,restart+75)]=1
                events.append(restart)
            else:
                preparation[a:b]=np.maximum(preparation[a:b],slow[a:b]*still[a:b])
    return dict(slow=slow,still=still,preparation=preparation,after_restart=after_restart,setup=setup,restarts=np.array(events))


def augment_emissions(e,contact,f,c):
    out=e.copy();diagnostic={}
    if c.get('stoppage_mode'):
        stop=stoppage_features(c.get('kick_speed',2.5),c.get('walking_speed',1.5));mode=c['stoppage_mode']
        if mode=='walking_still':cue=stop['slow']*stop['still']
        elif mode=='tag_walking':cue=stop['setup']*stop['slow']
        elif mode=='restart':cue=stop['preparation']-c.get('restart_live',.5)*stop['after_restart']
        else:raise ValueError('Unknown stoppage mode')
        out[:,2]+=c.get('stoppage_weight',1)*cue
        # During prepared restarts a close player need not have live possession.
        if c.get('suppress_setup_contact',False):out[:,:2]*=(1-.7*stop['preparation'][:,None])
        diagnostic['inferred_restarts']=stop['restarts']
    if c.get('homography_mode'):
        h=homography_features();trust=1/(1+(h['error']/c.get('homography_scale',.5))**2);trust*=h['valid']
        mode=c['homography_mode']
        if mode=='reliability':out*=trust[:,None]
        elif mode=='team':out[:,:2]*=trust[:,None]
        elif mode=='image':
            factor=np.where(f['original_seen']&~h['in_image'],.2,1.)*trust
            out[:,:2]*=factor[:,None]
        elif mode=='visibility':
            mask=~f['original_seen'];out[mask,2]*=np.minimum(1,h['field_fraction'][mask]/.5)
        else:raise ValueError('Unknown homography mode')
    if c.get('broadcast_weight'):
        from possession.cameras import aligned_broadcast
        b=aligned_broadcast();d=b['distance'];ok=b['seen'].copy()
        mode=c.get('broadcast_gate','missing')
        if mode=='missing':ok &= ~f['original_seen']
        elif mode=='ambiguous':ok &= ~contact.any(1)
        elif mode!='all':raise ValueError('Unknown broadcast gate')
        # Independent-camera control only when very close and spatially separated
        # from the opponent. Do not infer D when broadcast loses the ball.
        ctl=(d<c.get('broadcast_radius',1.5)) & ((d[:,::-1]-d)>.5) & ok[:,None]
        signal=ctl.astype(float)-ctl[:,::-1].astype(float)
        quality=np.clip(1-b['geometry_error']/3,0,1)
        out[:,:2]+=c['broadcast_weight']*signal*quality[:,None]
        out[:,2]-=c.get('broadcast_live',1)*ctl.any(1)*quality
    return out,diagnostic

