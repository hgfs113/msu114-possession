"""Soft whistle-conditioned activity evidence and temporal decoding."""
import numpy as np


def audio_signals(events, centres, minimum_score=.7):
    near = np.zeros(len(centres))
    recent = np.zeros(len(centres))
    for event in events:
        if event['score'] < minimum_score:
            continue
        delta = centres - event['start_seconds']
        weight = float(event['score'])
        near = np.maximum(near, weight*np.exp(-.5*(delta/.65)**2)*(np.abs(delta)<2))
        recent = np.maximum(recent, weight*np.exp(-np.maximum(delta, 0)/2)*((delta>=0)&(delta<6)))
    return near, recent


def context_score(x, names):
    f = {name:x[:, i] for i, name in enumerate(names)}
    return np.clip(.5*f['walking']+.45*f['tempo_drop']+.25*f['still']+
                   .3*f['missing']+.3*f['setpiece']-.55*f['ball_motion']-
                   .3*f['running']-.4*f['after_restart'], -1, 1)


def decode(score, enter=15, leave=15, boundary=None, strength=0, context=None):
    """Binary HMM with optional time-varying transition costs, per half."""
    score = np.asarray(score, dtype=float)
    if not len(score):
        return np.empty(0, dtype='i1')
    boundary = np.zeros(len(score)) if boundary is None else np.asarray(boundary)
    state_score = np.zeros(2)
    back = np.empty((len(score), 2), dtype='i1')
    for t, value in enumerate(score):
        pulse = np.clip(boundary[t]*strength, 0, .95)
        stop_gate = restart_gate = 1.
        if context is not None:
            stop_gate = .25+.75*max(0, context[t])
            restart_gate = .25+.75*max(0, -context[t])
        transition = np.array([[0., -enter*(1-pulse*stop_gate)],
                               [-leave*(1-pulse*restart_gate), 0.]])
        transition -= np.logaddexp.reduce(transition, axis=1)[:, None]
        paths = state_score[:, None] + transition
        index = paths.argmax(0)
        back[t] = index
        state_score = paths[index, np.arange(2)] + np.array([0., value])
        state_score -= state_score.max()
    result = np.empty(len(score), dtype='i1')
    state = int(state_score.argmax())
    for t in range(len(score)-1, -1, -1):
        result[t] = state
        state = int(back[t, state])
    return result
