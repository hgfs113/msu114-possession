from pathlib import Path


import hashlib,json


import numpy as np


from possession.preprocess import HALVES,PITCH,fix_goalkeepers


from possession.base import CACHE,N,inputs,emissions,viterbi


def kinematic_filter(xy,noise=.35,process=12.,gap=.6,adaptive=False,rts=False,halves=HALVES):
    """CV Kalman with continuous acceleration noise, optional RTS smoothing.

    Returns positions/velocities in metres, position variance, valid observations.
    RTS uses full covariance and never bridges a reset, long gap or half boundary.
    """
    z=np.asarray(xy,dtype=float)*PITCH
    if z.ndim==2:z=z[:,None,:]
    n,k,_=z.shape;pos=np.full((n,k,2),np.nan,dtype='f4');vel=pos.copy();variance=np.full((n,k),np.nan,dtype='f4')
    observed=np.isfinite(z).all(2);dt=.04;r=noise**2
    for a,b in halves:
        a=max(0,a);b=min(n,b)
        if a>=b:continue
        length=b-a;p=np.zeros((k,2));v=p.copy();c00=np.ones(k)*r;c01=np.zeros(k);c11=np.ones(k)*100
        active=np.zeros(k,bool);age=np.full(k,100000);reset_history=np.zeros((length,k),bool)
        covariance=np.empty((length,k,3),dtype='f4') if rts else None
        qhistory=np.empty((length,k),dtype='f4') if rts else None
        for t in range(a,b):
            seen=observed[t];p+=dt*v;age+=1
            residual=np.where(seen[:,None],z[t]-p,0.)
            q=np.full(k,process**2)
            if adaptive:q*=np.clip(np.sum(residual**2,axis=1)/(2*(c00+r)),1,40)
            c00+=2*dt*c01+dt*dt*c11+q*dt**3/3;c01+=dt*c11+q*dt*dt/2;c11+=q*dt
            reset=seen&(~active|(age>gap*25+1)|(np.linalg.norm(residual,axis=1)>7))
            p[reset]=z[t,reset];v[reset]=0;c00[reset]=r;c01[reset]=0;c11[reset]=100
            update=seen&~reset;s=c00+r;k0=c00/s;k1=c01/s
            p[update]+=k0[update,None]*residual[update];v[update]+=k1[update,None]*residual[update]
            old00=c00.copy();old01=c01.copy();c00[update]-=k0[update]*old00[update];c01[update]-=k0[update]*old01[update];c11[update]-=k1[update]*old01[update]
            age[seen]=0;active|=seen;valid=active&(age<=gap*25)
            pos[t,valid]=p[valid];vel[t,valid]=v[valid];variance[t,valid]=c00[valid]
            reset_history[t-a]=reset
            if rts:covariance[t-a]=np.c_[c00,c01,c11];qhistory[t-a]=q
        if rts:
            # 2x2 closed-form smoother gain for each independent coordinate axis.
            next_cov=covariance[-1].astype(float)
            for t in range(b-2,a-1,-1):
                valid=np.isfinite(pos[t]).all(1)&np.isfinite(pos[t+1]).all(1)&~reset_history[t+1-a]
                c00,c01,c11=covariance[t-a].T.astype(float);q=qhistory[t+1-a]
                pp00=c00+2*dt*c01+dt*dt*c11+q*dt**3/3;pp01=c01+dt*c11+q*dt*dt/2;pp11=c11+q*dt
                det=np.maximum(pp00*pp11-pp01**2,1e-12)
                g00=((c00+dt*c01)*pp11-c01*pp01)/det;g01=(-(c00+dt*c01)*pp01+c01*pp00)/det
                g10=((c01+dt*c11)*pp11-c11*pp01)/det;g11=(-(c01+dt*c11)*pp01+c11*pp00)/det
                dp=pos[t+1]-(pos[t]+dt*vel[t]);dv=vel[t+1]-vel[t]
                pos[t,valid]+=g00[valid,None]*dp[valid]+g01[valid,None]*dv[valid]
                vel[t,valid]+=g10[valid,None]*dp[valid]+g11[valid,None]*dv[valid]
                d00=next_cov[:,0]-pp00;d01=next_cov[:,1]-pp01;d11=next_cov[:,2]-pp11
                s00=c00+g00*g00*d00+2*g00*g01*d01+g01*g01*d11
                s01=c01+g00*g10*d00+(g00*g11+g01*g10)*d01+g01*g11*d11
                s11=c11+g10*g10*d00+2*g10*g11*d01+g11*g11*d11
                next_cov=covariance[t-a].astype(float);next_cov[valid]=np.c_[s00,s01,s11][valid]
                variance[t,valid]=np.maximum(s00[valid],0)
    return pos,vel,variance,observed


def kinematics(kind):
    # Make sure organiser cache/signature is current, then cache only small derived features.
    base=inputs('fixed');signature=(CACHE/'source_signature.json').read_text()
    key=hashlib.sha256((signature+'contact-v1-'+kind).encode()).hexdigest()[:12];path=CACHE/f'contact_{kind}_{key}.npz'
    if path.exists():return dict(np.load(path))
    z=np.load(CACHE/'tracking_raw.npz');xy,_=fix_goalkeepers(z['xy']);ball=z['ball'];teams=z['teams']
    if kind=='raw':
        allpos=np.concatenate([xy,ball[:,None]],axis=1)*PITCH;observed=np.isfinite(allpos).all(2)
        allvel=np.full_like(allpos,np.nan)
        for a,b in HALVES:
            # Symmetric 0.16-second finite difference; never cross a track gap.
            for offset in [2]:
                allvel[a+offset:b-offset]=(allpos[a+2*offset:b]-allpos[a:b-2*offset])/(2*offset*.04)
        var=np.ones(observed.shape)*.16
    else:
        settings={'split_rts':(.35,12.,False,True),'fast_ball_rts':(.2,30.,True,True),'stiff_rts':(.1,4.,False,True),'kf':(.35,12.,False,False),'adaptive':(.35,8.,True,False),'rts':(.35,12.,False,True),'adaptive_rts':(.35,8.,True,True),'tight_rts':(.15,18.,True,True),'soft_rts':(.7,8.,True,True)}
        noise,process,adaptive,rts=settings[kind]
        if kind in ('split_rts','fast_ball_rts','stiff_rts'):process=np.r_[np.full(xy.shape[1],process/6),process]
        # Joint loop, but no physical coupling or pseudo-measurement to nearby players.
        allpos,allvel,var,observed=kinematic_filter(np.concatenate([xy,ball[:,None]],axis=1),noise,process,.6,adaptive,rts)
    d=np.linalg.norm(allpos[:,:-1]-allpos[:,-1,None],axis=2);d[~np.isfinite(d)]=1e6
    relative=np.linalg.norm(allvel[:,:-1]-allvel[:,-1,None],axis=2);relative[~np.isfinite(relative)]=1e3
    distance=np.zeros((N,2));relv=distance.copy();unc=distance.copy();slots=np.zeros((N,2),dtype='i2');actual=np.zeros((N,2),bool)
    for t in (0,1):
        ids=np.flatnonzero(teams==t);best=ids[d[:,ids].argmin(1)];idx=np.arange(N)
        slots[:,t]=best;distance[:,t]=d[idx,best];relv[:,t]=relative[idx,best];unc[:,t]=var[idx,best]+var[:,-1]
        actual[:,t]=observed[idx,best]&observed[:,-1]
    speed=np.linalg.norm(allvel[:,-1],axis=1);speed=np.nan_to_num(speed,nan=0)
    f=dict(distance=distance,relv=relv,uncertainty=np.nan_to_num(unc,nan=100),slots=slots,actual=actual,speed=speed,seen=np.isfinite(allpos[:,-1]).all(1))
    for key in ('nearest_gap','gap','outside','original_seen','tags','inphase','restart','valid','close_count'):f[key]=base[key]
    np.savez_compressed(path,**f);return f


def sustained_contact(f,c):
    d=f['distance'];radius=c.get('radius',1.25);margin=c.get('margin',.5)
    contact=(d<=radius)&((d[:,::-1]-d)>=margin)&f['actual']
    if c.get('relative_speed',0):contact&=f['relv']<=c['relative_speed']
    if c.get('max_uncertainty',0):contact&=f['uncertainty']<=c['max_uncertainty']**2
    minimum=c.get('contact_frames',3);events=np.zeros_like(contact)
    for a,b in HALVES:
        for team in (0,1):
            # Stable player slot within each event; no assumed identity across gaps.
            mark=contact[a:b,team];slot=f['slots'][a:b,team]
            cuts=np.r_[0,np.flatnonzero((mark[1:]!=mark[:-1])|(slot[1:]!=slot[:-1]))+1,b-a]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if mark[lo] and hi-lo>=minimum:events[a+lo:a+hi,team]=True
    return events


def carry_events(contact,seconds):
    """Decayed last observed control; never manufacture contacts from extrapolation."""
    out=np.zeros((N,2))
    for a,b in HALVES:
        owner=-1;age=1e6
        for i in range(a,b):
            if contact[i].any():owner=int(contact[i].argmax());age=0
            else:age+=1
            if owner>=0:out[i,owner]=np.exp(-age/(25*seconds))
    return out


def decode_general(e,transition,classes):
    k=e.shape[1];trans=transition.copy();high=trans.max(1);trans-=(high+np.log(np.exp(trans-high[:,None]).sum(1)))[:,None]
    score=np.zeros(k);back=np.empty((len(e),k),dtype='i2')
    for i,row in enumerate(e):
        paths=score[:,None]+trans;j=paths.argmax(0);back[i]=j;score=paths[j,np.arange(k)]+row;score-=score.max()
    out=np.empty(len(e),dtype='i1');j=score.argmax()
    for i in range(len(e)-1,-1,-1):out[i]=classes[j];j=back[i,j]
    return out


def contact_predict(c,return_contacts=False,return_emissions=False):
    f=kinematics(c.get('kinematics','adaptive_rts'));contact=sustained_contact(f,c)
    cfg={**c,'flight':c.get('flight',8),'gap':c.get('gap',1),'tags':0}
    geometry=inputs('fixed') if c.get('base_geometry')=='fixed' else f
    if c.get('kalman_speed',False):geometry={**geometry,'speed':f['speed']}
    base=emissions(geometry,cfg);team=base[:,:2]*c.get('geometry_weight',.3)
    strength=c.get('contact_weight',4);team+=strength*(contact.astype(float)-contact[:,::-1].astype(float))
    if c.get('memory',0):
        memory=carry_events(contact,c['memory']);team+=c.get('memory_weight',.7)*(memory-memory[:,::-1])
    dead=base[:,2].copy()
    if c.get('stationary_dead',0):
        for a,b in HALVES:
            still=f['original_seen'][a:b]&(f['speed'][a:b]<c.get('stationary_speed',.5))
            cuts=np.r_[0,np.flatnonzero(still[1:]!=still[:-1])+1,b-a]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if still[lo] and hi-lo>=c.get('stationary_frames',50):dead[a+lo:a+hi]+=c['stationary_dead']
    e=np.c_[team,dead]
    if return_emissions:return e,contact,f
    out=np.full(N,'X',dtype='<U1');step=c.get('step',5)
    for a,b in HALVES:
        red=np.array([e[i:min(i+step,b)].mean(0) for i in range(a,b,step)])
        if c.get('window',0):
            w=c['window'];ix=np.arange(len(red));lo=np.maximum(0,ix-w);hi=np.minimum(len(red),ix+w+1);cs=np.vstack([np.zeros(3),np.cumsum(red,axis=0)]);red=(cs[hi]-cs[lo])/(hi-lo)[:,None]
        if c.get('factorized',False):
            cost=c.get('switch',5);tt=np.array([[0.,-cost],[-cost,0.]])
            team_state=decode_general(red[:,:2],tt,np.array([0,1]))
            dead_cost=c.get('dead_switch',8);at=np.array([[0.,-dead_cost],[-dead_cost,0.]])
            alive=np.c_[red[:,:2].max(1)+c.get('alive_bias',0),red[:,2]]
            live_state=decode_general(alive,at,np.array([0,1]));states=np.where(live_state==0,team_state,2)
        elif c.get('states',3)==3:states=viterbi(red,c.get('switch',5),c.get('dead_switch',8),c.get('dwell',1))
        else:
            # H-control / H-flight / A-control / A-flight / dead. Flight keeps
            # the releasing team; direct switches between flight states are forbidden.
            hits=np.array([contact[i:min(i+step,b)].mean(0) for i in range(a,b,step)])
            obs=np.c_[red[:,0]-1+hits[:,0],red[:,0]*.25-hits.sum(1),red[:,1]-1+hits[:,1],red[:,1]*.25-hits.sum(1),red[:,2]]
            trans=np.full((5,5),-1e6);np.fill_diagonal(trans,0)
            for ctl,flight,other in [(0,1,2),(2,3,0)]:
                trans[ctl,flight]=-.5;trans[flight,ctl]=-.5;trans[ctl,other]=-c.get('switch',5);trans[flight,other]=-c.get('switch',5)
                trans[ctl,4]=trans[flight,4]=-c.get('dead_switch',8);trans[4,ctl]=-2;trans[4,flight]=-4
            states=decode_general(obs,trans,np.array([0,0,1,1,2]))
        out[a:b]=np.repeat(np.array(['H','A','D'])[states],step)[:b-a]
    return (out,contact,f) if return_contacts else out

