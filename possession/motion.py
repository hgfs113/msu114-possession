import numpy as np


from possession.base import inputs,CACHE,N


from possession.preprocess import HALVES,PITCH,fix_goalkeepers


def motion_features():
    inputs('fixed');path=CACHE/'motion_context_v2.npz'
    if path.exists() and path.stat().st_mtime_ns >= (CACHE/'tracking_raw.npz').stat().st_mtime_ns:return dict(np.load(path))
    raw=np.load(CACHE/'tracking_raw.npz');xy,_=fix_goalkeepers(raw['xy']);xy*=PITCH
    speed=np.full(xy.shape[:2],np.nan)
    for a,b in HALVES:speed[a+5:b-5]=np.linalg.norm(xy[a+10:b]-xy[a:b-10],axis=2)/.4
    speed[speed>12]=np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');median=np.nanmedian(speed,axis=1)
    out=dict(median_speed=np.nan_to_num(median,nan=0),moving=(speed>1.5).sum(1),observed_count=np.isfinite(speed).sum(1))
    np.savez_compressed(path,**out);return out

