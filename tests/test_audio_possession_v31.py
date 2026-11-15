import unittest
from unittest.mock import patch

import numpy as np

from possession.whistle.activity_v31 import half_events,signals,snap_boundaries,AudioPossession


NAMES=['walking','tempo_drop','still','missing','setpiece','ball_motion','running','after_restart','contact']


class AudioPossessionTests(unittest.TestCase):
    def test_intermission_cannot_leak_into_next_half(self):
        events=[dict(id='break',start_seconds=19.8,end_seconds=20.3,peak_seconds=19.9),
                dict(id='play',start_seconds=19.9,end_seconds=20.4,peak_seconds=20.1)]
        inside=half_events(events,500,750)
        self.assertEqual([e['id'] for e in inside],['play'])
        self.assertEqual(inside[0]['start_seconds'],20.)
        self.assertEqual(half_events(events,0,250),[])

    def test_rolling_ball_alone_does_not_authorise_restart(self):
        x=np.zeros((100,len(NAMES)));x[:,NAMES.index('ball_motion')]=1
        event=[dict(id='w',start_seconds=8.,score_db=25)]
        times=np.arange(100)/5;state=np.ones(100,dtype=int)
        permissive=signals(event,times,x,np.zeros(100),state,NAMES,dict(restart_policy='permissive'))
        strict=signals(event,times,x,np.zeros(100),state,NAMES,dict(restart_policy='strict'))
        self.assertLess(permissive['role'].min(),-.9);self.assertEqual(strict['role'].min(),0)
        x[:,NAMES.index('contact')]=1;x[:,NAMES.index('running')]=1
        clear=signals(event,times,x,np.zeros(100),state,NAMES,dict(restart_policy='strict'))
        self.assertLess(clear['role'].min(),-.9)

    def test_no_audio_preserves_control_and_outside_is_masked(self):
        model=AudioPossession.__new__(AudioPossession);model.events=[];model.names=NAMES
        model.config=dict(enter=15,leave=15)
        model.pack=dict(owner=np.zeros(30,dtype=int),x0=np.zeros((2,len(NAMES))),x1=np.zeros((2,len(NAMES))),
                        score0=np.array([-1.,-1.]),score1=np.array([1.,1.]),state0=np.zeros(2,int),state1=np.ones(2,int))
        with patch('possession.whistle.activity_v31.HALVES',((0,10),(20,30))),patch('possession.__main__.submission_labels',side_effect=lambda a:a):
            labels,activity,roles=model.predict(dict(mode='role',weight=16,restart_policy='strict'))
        self.assertEqual(''.join(labels),'H'*10+'X'*10+'D'*10)
        self.assertEqual(''.join(activity),'L'*10+'X'*10+'D'*10);self.assertEqual(roles,[])

    def test_boundary_snap_does_not_cross_an_existing_spell(self):
        states=np.array([0]*5+[1]*2+[0]*8);times=np.arange(15)/5
        result=snap_boundaries(states,[dict(start_seconds=2.)],times,2)
        # The first boundary is blocked by the second; an intervening spell is not erased.
        self.assertEqual(result[4],0);self.assertEqual(result[5],1)


if __name__=='__main__':unittest.main()
