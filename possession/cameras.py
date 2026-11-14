import hashlib


import json


from functools import lru_cache


from pathlib import Path


from types import SimpleNamespace


import numpy as np


from games import asset_paths


from possession.base import CACHE, N, inputs


from possession.preprocess import HALVES, PITCH, fix_goalkeepers


from possession.tracking import Source, build_sync, geometry_error


from starter.load_homography import load as load_homography


def signature():
    return {s:{k:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for k,p in asset_paths(source=s).items() if k!='video'} for s in ['tactical','broadcast']}


def sources():
    inputs('fixed')
    z=np.load(CACHE/'tracking_raw.npz');xy,_=fix_goalkeepers(z['xy'])
    tactical=SimpleNamespace(n=N,xy=xy,ball=z['ball'],teams=z['teams'],valid=z['valid'])
    key=hashlib.sha256(json.dumps(signature()['broadcast'],sort_keys=True).encode()).hexdigest()[:12]
    path=CACHE/f'broadcast_raw_{key}.npz'
    if not path.exists():
        source=Source('broadcast',correct_goalkeepers=False)
        np.savez_compressed(path,xy=source.xy,ball=source.ball,teams=source.teams,valid=source.valid)
    z=np.load(path);broadcast=SimpleNamespace(n=len(z['ball']),xy=z['xy'],ball=z['ball'],teams=z['teams'],valid=z['valid'])
    for source in [tactical,broadcast]:
        source.present=np.isfinite(source.xy).all(2);source.ball_seen=np.isfinite(source.ball).all(1)
    return tactical,broadcast


def bracketed_mapping(anchors,n,halves=HALVES,max_error=2,max_gap=250,max_offset_delta=20):
    """Require 3+ consecutive anchors with stable offset and orientation.

    Return a broadcast index/flip for tactical frames. Never extrapolate anchors.
    This is a candidate map; framewise homography/geometry checks come afterwards.
    """
    ordered=sorted([a for a in anchors if a['error']<=max_error],key=lambda a:a['b'])
    groups=[];group=[]
    for anchor in ordered:
        if group:
            last=group[-1];db=anchor['b']-last['b'];dt=anchor['t']-last['t']
            if not (0<db<=max_gap and 0<dt and abs(dt-db)<=max_offset_delta and anchor['flip']==last['flip']):
                groups.append(group);group=[]
        group.append(anchor)
    if group:groups.append(group)
    mapping=np.full(n,-1,dtype='i4');flips=np.zeros(n,dtype='i1');quality=np.full(n,np.inf)
    accepted=[]
    for group in groups:
        if len(group)<3 or group[-1]['b']-group[0]['b']<100:continue
        for left,right in zip(group[:-1],group[1:]):
            a,b=left['t'],right['t']+1
            if not any(lo<=a<b<=hi for lo,hi in halves):continue
            q=max(left['error'],right['error']);ix=np.arange(a,b)
            # Fractional interpolation accommodates local sync refinement, not video stretching.
            bi=np.rint(np.interp(ix,[a,b-1],[left['b'],right['b']])).astype(int)
            better=q<quality[ix];mapping[ix[better]]=bi[better];flips[ix[better]]=left['flip'];quality[ix[better]]=q
        accepted.append(group)
    return mapping,flips,quality,accepted


@lru_cache(maxsize=1)
def aligned_broadcast():
    inputs('fixed');key=hashlib.sha256((json.dumps(signature(),sort_keys=True)+'camera-v1').encode()).hexdigest()[:12]
    cache=CACHE/f'camera_features_{key}.npz'
    if cache.exists():return dict(np.load(cache))
    tactical,broadcast=sources();anchor_path=CACHE/f'camera_anchors_{key}.json'
    if not anchor_path.exists():
        anchors=build_sync(tactical,broadcast,{})
        anchor_path.write_text(json.dumps(anchors))
    else:anchors=json.loads(anchor_path.read_text())
    mapping,flips,quality,groups=bracketed_mapping(anchors,N)
    _,error,valid=load_homography(source='broadcast')
    available=mapping>=0;bi=np.maximum(mapping,0)
    available &= valid[bi] & (error[bi]<1.)
    # Test geometric consistency every 0.2s. Reject inconsistent observations,
    # including broadcast close-ups/replays that fall between otherwise good anchors.
    geometry=np.full(N,np.inf)
    for a,b in HALVES:
        for i in range(a,b,5):
            end=min(i+5,b)
            if not available[i]:continue
            e=geometry_error(tactical,broadcast,i,int(mapping[i]),int(flips[i]))
            geometry[i:end]=e
    available &= geometry<2.5
    bp=broadcast.ball[bi].copy();xy=broadcast.xy[bi].copy()
    for axis in (0,1):
        flip=(flips&(1<<axis))!=0;bp[flip,axis]=1-bp[flip,axis];xy[flip,:,axis]=1-xy[flip,:,axis]
    # Use slots only within this frame; cross-camera player IDs are never matched.
    dist=np.linalg.norm((xy-bp[:,None])*PITCH,axis=2);dist[~np.isfinite(dist)]=1e6
    distance=np.column_stack([dist[:,np.asarray(broadcast.teams)==team].min(1) for team in (0,1)])
    seen=available & np.isfinite(bp).all(1)
    both=seen & tactical.ball_seen;ball_error=np.full(N,np.nan)
    ball_error[both]=np.linalg.norm((bp[both]-tactical.ball[both])*PITCH,axis=1)
    out=dict(mapping=mapping,flip=flips,available=available,seen=seen,distance=distance,
             geometry_error=geometry,homography_error=error[bi],ball_error=ball_error)
    np.savez_compressed(cache,**out)
    diagnostic=dict(anchors=len(anchors),stable_groups=len(groups),bracketed_frames=int((mapping>=0).sum()),
                    geometry_valid_frames=int(available.sum()),ball_seen_frames=int(seen.sum()),
                    extra_ball_frames=int((seen & ~tactical.ball_seen).sum()),
                    simultaneous_ball_frames=int(both.sum()),
                    ball_distance_quantiles=np.nanquantile(ball_error,[.1,.5,.9]).tolist() if both.any() else [])
    (CACHE/'camera_alignment_summary.json').write_text(json.dumps(diagnostic,indent=2))
    return out

