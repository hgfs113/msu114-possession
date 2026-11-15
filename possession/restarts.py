"""Detect restarts and retire stale set-piece context; no reference inputs.

Offline detector: a candidate may use up to three seconds of subsequent measured
play for confirmation. It does not identify the team from a set-piece tag.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from possession.preprocess import HALVES, PITCH
from possession.ownership import smooth_signal
from possession.activity.features import reduce_blocks
from possession.activity.model import train_logistic, physical_seeds, moving_average
from possession.goalkeeper.phases import KeeperPhaseModel, phase_evidence
from possession.goalkeeper.model import COMMON
from possession.whistle.activity_v31 import half_events, signals
from possession.whistle.half_model import heuristic
from possession.whistle.fusion import decode
from possession.whistle.detect import sha

BASE_CONFIG = dict(mode='soft', anchor='carry', require_release=True, gate='context', weight=6)
DEFAULT_CACHE = Path('.cache')


def candidate_times(speed, seen, slow, still, moving, a, b):
    """Original candidate definition, shared by first/best/confirmed ablations."""
    candidates = []
    search = a + min(25, (b-a)//4)
    for t in range(search, b-10, 5):
        past = max(a, t-50)
        quiet = ((speed[past:t] < 1.) | (~seen[past:t])).mean()
        if moving[t] > .65 and quiet > .65:
            support = quiet + .4*slow[past:t].mean() + .2*still[past:t].mean()
            candidates.append((float(support), t))
    return candidates


def confirmation(data, t, a, b, detector, speed_threshold):
    """Reject ball relocation/jumps; require a measured pass or continued play."""
    seen = data['seen']; speed = data['speed']; ball = data['ball']
    lo = max(a, t-15); hi = min(b, t+75)
    at = np.arange(lo, hi, 5)
    valid = seen[at] & np.isfinite(ball[at]).all(1)
    at = at[valid]
    if len(at) < 4:
        return False, dict(reason='too_few_observations')
    times = at/25
    step_speed = np.linalg.norm(np.diff(ball[at], axis=0), axis=1)/np.diff(times)
    # Long gaps cannot confirm a coherent measured trajectory.
    coherent = (np.diff(times) <= .4) & (step_speed < 45)
    if coherent.sum() < 3 or np.any((np.diff(times) <= .4) & (step_speed >= 45)):
        return False, dict(reason='incoherent_trajectory')
    origin_index = int(np.argmin(abs(at-t)))
    origin = at[origin_index]
    if abs(origin-t) > 10:
        return False, dict(reason='missing_origin')
    # Displacement must belong to the connected measured trajectory containing
    # the launch, not two stationary clusters separated by a tracking gap.
    left = origin_index
    right = origin_index+1
    while left > 0 and coherent[left-1]:
        left -= 1
    while right < len(at) and coherent[right-1]:
        right += 1
    if right-left < 4:
        return False, dict(reason='short_connected_trajectory')
    displacement = float(np.max(np.linalg.norm(ball[at[left:right]]-ball[origin], axis=1)))
    post = slice(t, min(b, t+50))
    activity = float((seen[post] & (speed[post] > speed_threshold)).mean())
    touch = bool(data['contact'][lo:min(b,t+11)].any())
    receive = bool(data['contact'][min(b,t+15):hi].any())
    running = float(data['running'][post].mean())
    in_field = bool((-1.5 <= ball[origin,0] <= PITCH[0]+1.5) and
                    (-1.5 <= ball[origin,1] <= PITCH[1]+1.5))
    moving = displacement >= 3 and activity >= .35 and in_field
    accepted = moving and touch
    if detector == 'play':
        accepted = moving and (touch or receive or (activity >= .7 and running >= .25))
    return bool(accepted), dict(displacement=displacement, activity=activity,
                               touch=touch, receive=receive, running=running)


def tag_features(data, tags, detector='best', speed_threshold=4, retirement='legacy'):
    """Construct tag features per event, so overlapping tags retain their evidence."""
    n = len(data['speed']); seen = data['seen']; speed = data['speed']
    slow = data['slow']; still = data['still']
    moving = smooth_signal((seen & (speed > speed_threshold)).astype(float), .4)
    # The old tag features always used speed 4. Keep them unchanged before a new
    # confirmed boundary, instead of inventing additional preparation beforehand.
    old_moving = smooth_signal((seen & (speed > 4)).astype(float), .4)
    result = {k: np.zeros(n) for k in ('setpiece', 'preparation', 'after_restart', 'goal_tag')}
    events = []
    for event in tags:
        if event['code'] != 'SET PIECES':
            continue
        for h, (lo, hi) in enumerate(HALVES):
            a = max(lo, event['start_frame']); b = min(hi, event['end_frame'])
            if b <= a:
                continue
            old_candidates = candidate_times(speed, seen, slow, still, old_moving, a, b)
            old = max(old_candidates)[1] if old_candidates else None
            prep = np.zeros(b-a); restart_signal = np.zeros(b-a)
            if old is not None:
                prep[:old-a] = .4 + .6*slow[a:old]
                restart_signal[old-a:min(b,old+75)-a] = 1
            else:
                prep[:] = slow[a:b]*still[a:b]
            candidates = candidate_times(speed, seen, slow, still, moving, a, b)
            selected = None; evidence = {}; rejected = 0
            if detector == 'best':
                selected = max(candidates)[1] if candidates else None
            else:
                for _, t in candidates:
                    good, evidence = (True, {}) if detector == 'first' else confirmation(data,t,a,b,detector,speed_threshold)
                    if good:
                        selected = t
                        break
                    rejected += 1
            setup = np.ones(b-a)
            if retirement != 'legacy' and selected is not None:
                prep[selected-a:] = 0
                if retirement != 'preparation':
                    setup[selected-a:] = 0
                    restart_signal[:] = 0
                    restart_signal[selected-a:min(b,selected+75)-a] = 1
            result['setpiece'][a:b] = np.maximum(result['setpiece'][a:b], setup)
            result['preparation'][a:b] = np.maximum(result['preparation'][a:b], prep)
            result['after_restart'][a:b] = np.maximum(result['after_restart'][a:b], restart_signal)
            if event['tags'].get('Type') == 'Goal Kick':
                result['goal_tag'][a:b] = np.maximum(result['goal_tag'][a:b],setup)
            events.append(dict(half=h+1, start=a, end=b, type=event['tags'].get('Type'),
                               old_restart=old, restart=selected, candidates=len(candidates),
                               rejected=rejected, evidence=evidence))
    return result, events


def prepare(output, cache=DEFAULT_CACHE, data_dir=None):
    from starter.load_smart_tagging import load, pattern_path
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    cache = Path(cache)
    files = {k:cache/v for k,v in dict(raw='tracking_raw.npz',kinematics='contact_adaptive_rts_b562702fc402.npz',motion='motion_context_v2.npz').items()}
    candidates=list(cache.glob('contact_adaptive_rts_*.npz'))
    if len(candidates)!=1: raise ValueError('Expected one prepared adaptive RTS cache')
    files['kinematics']=candidates[0]
    raw = np.load(files['raw']); k = np.load(files['kinematics']); motion = np.load(files['motion'])
    seen = k['original_seen'].astype(bool)
    players = motion['observed_count'] >= 6
    data = dict(ball=raw['ball']*PITCH, speed=k['speed'], seen=seen,
                slow=smooth_signal(((motion['median_speed']<1.8)&players).astype(float),1),
                still=smooth_signal((seen&(k['speed']<.5)).astype(float),1),
                contact=seen&np.any(k['actual']&(k['distance']<2.5),axis=1),
                running=np.divide(motion['moving'],motion['observed_count'],out=np.zeros(len(seen)),where=players))
    tags = load(data_dir=data_dir)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(dict(pack_sha256=sha(output),
        inputs={str(p):sha(p) for p in files.values()}, tag_path=str(pattern_path(data_dir=data_dir)),
        tag_sha256=sha(pattern_path(data_dir=data_dir)), tags=tags, uses_references=False),indent=2)+'\n')


class RestartModel:
    def __init__(self, path='outputs/restart/features.npz', base_config=None, physical_pack='outputs/audio_possession_v31/physical_features.npz', keeper_features='outputs/keeper_phase/features.npz', detections='outputs/whistles_v31/detections.json'):
        self.parent = KeeperPhaseModel(keeper_features,physical_pack,detections); self.pack = self.parent.pack
        self.names = self.pack['names'].tolist(); self.base_config = dict(base_config or BASE_CONFIG)
        path = Path(path); self.meta = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != self.meta['pack_sha256']:
            raise ValueError('Restart features changed')
        self.data = dict(np.load(path,allow_pickle=False)); self.tags = self.meta['tags']
        self.events = json.loads(Path(detections).read_text())['events']
        x = np.vstack([self.pack['x0'],self.pack['x1']]); chosen,y = physical_seeds(x)
        self.fit = train_logistic(x[chosen],y,l2=.1)
        self.cache = {}; self.baseline_error = {}
        legacy,_ = tag_features(self.data,self.tags)
        for h in (0,1):
            x = self.pack[f'x{h}']; reconstructed = moving_average(((x-self.fit['mean'])/self.fit['scale'])@self.fit['weight']+self.fit['intercept'],3)
            self.baseline_error[f'score{h}'] = float(abs(reconstructed-self.pack[f'score{h}']).max())
        for key in ('setpiece','preparation','after_restart'):
            blocks = reduce_blocks(smooth_signal(legacy[key],1))
            for h in (0,1):
                err=float(abs(blocks[h]-self.pack[f'x{h}'][:,self.names.index(key)]).max())
                self.baseline_error[f'{key}{h}']=err
        if max(self.baseline_error.values()) > 1e-6:
            raise ValueError(f'Legacy feature reconstruction failed: {self.baseline_error}')

    def predict(self,c):
        from possession.__main__ import submission_labels
        if c['detector']=='control':
            labels,_,_=self.parent.predict(self.base_config)
            return labels,[]
        key=json.dumps({k:v for k,v in c.items() if k!='name'},sort_keys=True)
        if key not in self.cache:
            tags,events=tag_features(self.data,self.tags,c['detector'],c['speed'],c['retirement'])
            blocks={k:reduce_blocks(smooth_signal(tags[k],1)) for k in ('setpiece','preparation','after_restart')}
            self.cache[key]=(tags,blocks,events)
        tags,blocks,events=self.cache[key]
        labels=np.full(len(self.pack['owner']),'X',dtype='<U1')
        for h,(a,b) in enumerate(HALVES):
            x=self.pack[f'x{h}'].copy()
            for k in blocks:x[:,self.names.index(k)]=blocks[k][h]
            physical=moving_average(((x-self.fit['mean'])/self.fit['scale'])@self.fit['weight']+self.fit['intercept'],3)
            audio=self.parent.base[h]['audio']
            if c['retirement']=='context':
                audio=signals(half_events(self.events,a,b),(np.arange(a,b,5)+2)/25,x,physical,decode(physical,15,15),self.names,COMMON)
            score=physical+2*heuristic(x,self.names,'tags')+16*audio['role']
            phase_features=self.parent.features
            phase_audio=self.parent.base[h]['audio']; prior=self.parent.base[h]['dead']==0
            if c['retirement']=='context':
                phase_features=dict(phase_features)
                idx=np.minimum(np.arange(a,b,5)+2,b-1)
                phase_features[f'goal_tag{h}']=tags['goal_tag'][idx]
                phase_audio=audio; prior=decode(score,15,15)==0
            p=phase_evidence(phase_features,h,prior,phase_audio,self.base_config)
            support=np.maximum(p['receive'],p['hold'])
            score-=self.base_config['weight']*support.max(1)
            score+=self.base_config['weight']*.5*p['setup'].max(1)
            dead=decode(score,15,15)
            owner=self.pack['owner'][a:b:5].copy();owner[dead==1]=2
            take=(support.max(1)>.5)&(dead==0);team=support.argmax(1);owner[take]=team[take]
            labels[a:b]=np.repeat(np.array(list('HADX'))[owner],5)[:b-a]
        return submission_labels(labels),events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--data-dir',type=Path)
    p.add_argument('--features',type=Path,default=Path('outputs/restart/features.npz'))
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare:
        prepare(a.features,data_dir=a.data_dir);return
    if not(a.config and a.output):p.error('--config and --output are required')
    c=json.loads(a.config.read_text());model=RestartModel(a.features,c.pop('base_config',None))
    labels,events=model.predict(c)
    from possession.base import write_prediction
    a.output.parent.mkdir(parents=True,exist_ok=True);write_prediction(a.output,labels)
    print(json.dumps(dict(output=str(a.output),restarts=sum(e['restart'] is not None for e in events),uses_references=False)))


if __name__=='__main__':main()
