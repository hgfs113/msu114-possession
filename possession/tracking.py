"""Local tracking loader and conservative geometric camera correspondences."""
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from games import asset_paths, GAMES, GAME, source_frames
FPS = 25
PITCH = np.array(GAMES[GAME]['pitch'], dtype=np.float32)
def derive(xy):
    present = np.isfinite(xy).all(axis=2)
    count = present.sum(axis=1)
    count_changed = np.r_[False, count[1:] != count[:-1]]
    set_changed = np.r_[False, np.any(present[1:] != present[:-1], axis=1)]
    distance = np.linalg.norm(np.diff(xy, axis=0) * PITCH, axis=2)
    jumps = np.vstack([np.zeros((1, xy.shape[1]), bool),
                       (distance > 5.0) & present[1:] & present[:-1]])
    return present, count, {'count': np.flatnonzero(count_changed).tolist(),
                            'set': np.flatnonzero(set_changed).tolist(),
                            'jump': np.flatnonzero(jumps.any(axis=1)).tolist()}, jumps


class Source:
    def __init__(self, name, demo=False, correct_goalkeepers=True):
        self.name = name
        if demo:
            self.n, slots = 3000, 22
            self.teams = [0]*11 + [1]*11
            self.names = ['Home demo', 'Away demo']
            t = np.arange(self.n)[:, None]
            ids = np.arange(slots)[None, :]
            self.xy = np.stack([np.broadcast_to(.12 + ids / 30, (self.n, slots)) + .04*np.sin(t/90),
                                np.broadcast_to(.1 + (ids % 7)/9, (self.n, slots)) + .03*np.cos(t/70)], axis=2).astype('f4')
            self.ball = np.stack([.5+.22*np.sin(t[:, 0]/80), .5+.2*np.cos(t[:, 0]/100)], axis=1).astype('f4')
            self.xy[200:230, 2] = np.nan
            self.xy[500, 6, 0] += .1
            self.ball[300:350] = np.nan
            self.valid = np.ones(self.n, bool)
        else:
            paths = asset_paths(source=name)
            meta = ET.parse(paths['atd_meta']).getroot()
            teams = list(meta.iter('Team'))
            self.names = [t.findtext('Name') or t.get('id') for t in teams]
            self.teams = [0 if p.get('teamId') == teams[0].get('id') else 1 for p in meta.iter('Player')]
            self.n = source_frames(name)
            self.xy = np.full((self.n, len(self.teams), 2), np.nan, dtype='f4')
            self.ball = np.full((self.n, 2), np.nan, dtype='f4')
            seen_ids = np.zeros(self.n, bool)
            with open(paths['atd_raw']) as f:
                for line in f:
                    frame, players, ball = line.strip().split(':')
                    i = int(frame)-1
                    if not 0 <= i < self.n or seen_ids[i]:
                        raise ValueError('Invalid or duplicate ATD frame ID')
                    seen_ids[i] = True
                    groups = players.split(';')
                    if len(groups) != len(self.teams):
                        raise ValueError('ATD slot count disagrees with metadata')
                    self.xy[i] = [[float(v) for v in g.split(',')[:2]] for g in groups]
                    self.ball[i] = [float(v) for v in ball.split(',')[:2]]
            if not seen_ids.all():
                raise ValueError('ATD frame IDs are not contiguous')
            self.valid = np.ones(self.n, bool)
            if Path(paths['homography']).exists():
                self.valid[:] = False
                with open(paths['homography']) as f:
                    for row in csv.DictReader(f):
                        i = int(row['frame'])
                        if 0 <= i < self.n:
                            self.valid[i] = row['valid'].strip() in ('1', 'True', 'true', '1.0')
        self.keeper_corrected = np.zeros(self.xy.shape[:2], bool)
        if name == 'tactical' and not demo and correct_goalkeepers:
            from possession.preprocess import fix_goalkeepers
            self.xy, self.keeper_corrected = fix_goalkeepers(self.xy)
        self.present, self.count, self.events, self.jumps = derive(self.xy)
        self.ball_seen = np.isfinite(self.ball).all(axis=1)


def features(source, ids):
    """Team shape descriptors, invariant to track identities; units are metres."""
    out = []
    for team in (0, 1):
        a = source.xy[ids][:, np.array(source.teams) == team] * PITCH
        # Only computed on frames where both teams have enough visible tracks.
        out.append(np.nanquantile(a, [.2, .5, .8], axis=1).transpose(1, 0, 2).reshape(len(ids), -1))
    return np.concatenate(out, axis=1)


def geometry_error(t, b, ti, bi, flip=0):
    """Symmetric nearest-neighbour distance, robust to missing/duplicated tracks."""
    errors = []
    for team in (0, 1):
        a = t.xy[ti, np.array(t.teams) == team]
        c = b.xy[bi, np.array(b.teams) == team].copy()
        a = a[np.isfinite(a).all(axis=1)]
        c = c[np.isfinite(c).all(axis=1)]
        if len(a) < 5 or len(c) < 5:
            return float('inf')
        if flip & 1: c[:, 0] = 1-c[:, 0]
        if flip & 2: c[:, 1] = 1-c[:, 1]
        d = np.linalg.norm((a[:, None]-c[None]) * PITCH, axis=2)
        errors.extend([float(np.median(d.min(axis=0))), float(np.median(d.min(axis=1)))])
    score = float(np.mean(errors))
    if t.ball_seen[ti] and b.ball_seen[bi]:
        ball = b.ball[bi].copy()
        if flip & 1: ball[0] = 1-ball[0]
        if flip & 2: ball[1] = 1-ball[1]
        score += .15*min(30, np.linalg.norm((t.ball[ti]-ball)*PITCH))
    return float(score)


def eligible(source):
    ids = np.arange(0, source.n, FPS*2)
    good = source.valid[ids].copy()
    for team in (0, 1):
        good &= source.present[ids][:, np.array(source.teams) == team].sum(axis=1) >= 7
    return ids[good]


def build_sync(t, b, status):
    """Conservative local correspondences, not a claim of official synchronization.

    Search independent snapshots; then require neighbouring matches to support the
    same time offset. Do not bridge edits, replays, or long unobserved stretches.
    """
    ti, bi = eligible(t), eligible(b)
    if not len(ti) or not len(bi): return []
    tf, bf = features(t, ti), features(b, bi)
    candidates = [[] for _ in bi]
    for flip in range(4):
        f = bf.copy().reshape(len(bi), 2, 3, 2)
        for axis in range(2):
            if flip & (1 << axis):
                f[..., axis] = PITCH[axis] - f[:, :, ::-1, axis]
        f = f.reshape(len(bi), -1)
        for start in range(0, len(bi), 64):
            q = f[start:start+64]
            distance = ((q[:, None]-tf[None])**2).mean(axis=2)
            k = min(3, len(ti))
            nearest = np.argpartition(distance, k-1, axis=1)[:, :k]
            for row, choices in enumerate(nearest):
                j = start+row
                for idx in choices:
                    err = geometry_error(t, b, int(ti[idx]), int(bi[j]), flip)
                    if err < 4:
                        candidates[j].append((float(err), int(ti[idx]), flip))
            status['progress'] = round(100*(flip+min(start+64,len(bi))/len(bi))/4)
    anchors = []
    for j, options in enumerate(candidates):
        ranked = []
        for err, frame, flip in options:
            offset = frame-int(bi[j])
            support = 0
            for z in range(max(0,j-4), min(len(bi),j+5)):
                if z == j or abs(int(bi[z])-int(bi[j])) > 250: continue
                if any(f == flip and abs((x-int(bi[z]))-offset) <= 50 for e,x,f in candidates[z]):
                    support += 1
            if support >= 3:
                ranked.append((err-.2*support, err, frame, flip, support))
        if not ranked: continue
        ranked.sort()
        _, err, frame, flip, support = ranked[0]
        # Refine the sampled tactical position at the same broadcast frame.
        choices = range(max(0,frame-50), min(t.n,frame+51), 2)
        refined = min(choices, key=lambda x: geometry_error(t,b,x,int(bi[j]),flip))
        err = geometry_error(t,b,refined,int(bi[j]),flip)
        if err <= 3:
            anchors.append(dict(t=refined,b=int(bi[j]),error=round(err,3),flip=flip,support=support))
    return sorted(anchors, key=lambda x:x['t'])

