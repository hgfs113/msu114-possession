import numpy as np


HALVES=((11185,90652),(113742,190198))


PITCH=np.array([105.3,68.0])


def fix_goalkeepers(xy,halves=HALVES):
    """Mirror x only when a keeper appears in the opposite half.

    This is a geometric hypothesis, NOT recovered measured truth. Do not clamp
    legitimate advances into the penalty box; missing observations stay missing.
    The fixed goalkeeper-role configuration is specific to this match.
    """
    out=xy.copy();changed=np.zeros(xy.shape[:2],bool)
    if xy.shape[1]<46:return out,changed
    for half,(a,b) in enumerate(halves):
        for slot,left in [(23,half==0),(45,half==1)]:
            x=out[a:b,slot,0];bad=np.isfinite(out[a:b,slot]).all(1)&((x>.5) if left else (x<.5))
            x[bad]=1-x[bad];changed[a:b,slot]=bad
    return out,changed


def kalman(xy,measurement=.4,ball=False,max_gap=.8):
    """Constant velocity Kalman; bounded missing-data extrapolation; reset jumps.

    Independent x/y axes share scalar covariance; physical units are metres.
    No interpolation across halves or tracklet discontinuities. Output at 25 Hz.
    """
    z=np.asarray(xy,dtype=float)*PITCH
    if z.ndim==2:z=z[:,None,:]
    out=np.full_like(z,np.nan);dt=.04;q=(18. if ball else 5.)**2;r=measurement**2
    for a,b in HALVES:
        k=z.shape[1];pos=np.zeros((k,2));vel=np.zeros((k,2));p00=np.ones(k)*r;p01=np.zeros(k);p11=np.ones(k)*100
        age=np.full(k,10000);active=np.zeros(k,bool)
        for t in range(a,b):
            obs=z[t];seen=np.isfinite(obs).all(1);pos+=dt*vel
            p00+=2*dt*p01+dt*dt*p11+q*dt**4/4;p01+=dt*p11+q*dt**3/2;p11+=q*dt*dt
            age+=1;res=np.where(seen[:,None],obs-pos,0);jump=np.linalg.norm(res,axis=1)>(max(7,measurement*8) if ball else 5)
            reset=seen&(~active|(age>max_gap*25+1)|jump)
            pos[reset]=obs[reset];vel[reset]=0;p00[reset]=r;p01[reset]=0;p11[reset]=100
            update=seen&~reset;innovation=p00+r;k0=p00/innovation;k1=p01/innovation
            pos[update]+=k0[update,None]*res[update];vel[update]+=k1[update,None]*res[update]
            old00=p00.copy();old01=p01.copy();p00[update]-=k0[update]*old00[update];p01[update]-=k0[update]*old01[update];p11[update]-=k1[update]*old01[update]
            age[seen]=0;active|=seen;valid=active&(age<=max_gap*25);out[t,valid]=pos[valid]
    out/=PITCH
    return out[:,0] if xy.ndim==2 else out

