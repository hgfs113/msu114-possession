import unittest
from unittest.mock import patch

import numpy as np

from possession.restarts import confirmation, tag_features


class RestartTests(unittest.TestCase):
    def data(self,n=400):
        return dict(seen=np.ones(n,bool),speed=np.zeros(n),slow=np.ones(n),still=np.ones(n),
                    ball=np.tile([40.,34.],(n,1)),contact=np.zeros(n,bool),running=np.ones(n)*.4)

    def play(self,d,t,length=65):
        d['speed'][t:t+length]=8;d['still'][t:t+length]=0;d['contact'][t-5:t+5]=True
        d['ball'][t:t+length,0]=40+np.arange(length)*8/25
        d['ball'][t+length:,0]=d['ball'][t+length-1,0]

    def features(self,d,tags,**kw):
        # Small artificial half, without depending on match-specific timestamps.
        with patch('possession.restarts.HALVES',((0,len(d['seen'])),)),patch('possession.restarts.smooth_signal',side_effect=lambda x,s:x.astype(float)):
            return tag_features(d,tags,**kw)

    def test_pass_confirmed_but_relocation_and_large_jump_rejected(self):
        d=self.data();self.play(d,70)
        self.assertTrue(confirmation(d,70,0,250,'contact',4)[0])
        d['ball'][70:135,0]=40+np.arange(65)*.5/25;d['speed'][70:135]=.5
        self.assertFalse(confirmation(d,70,0,250,'contact',4)[0])
        d=self.data();self.play(d,70);d['ball'][80,0]=100
        self.assertFalse(confirmation(d,70,0,250,'contact',4)[0])

    def test_missing_ball_is_not_confirmation_and_cannot_use_next_half(self):
        d=self.data();self.play(d,70);d['seen'][:]=False
        self.assertFalse(confirmation(d,70,0,250,'play',4)[0])
        d=self.data();self.play(d,70)
        self.assertFalse(confirmation(d,70,0,75,'play',4)[0])

    def test_contact_and_play_distinguish_missing_kick_but_visible_continuation(self):
        d=self.data();self.play(d,70);d['contact'][:]=False
        self.assertFalse(confirmation(d,70,0,250,'contact',4)[0])
        self.assertTrue(confirmation(d,70,0,250,'play',4)[0])

    def test_first_false_motion_rejected_later_pass_retires_preparation(self):
        d=self.data();d['speed'][50:65]=8;self.play(d,140)
        tags=[dict(code='SET PIECES',start_frame=0,end_frame=300,tags={'Type':'Free Kick'})]
        raw,events=self.features(d,tags,detector='contact',retirement='tags')
        self.assertEqual(events[0]['restart'],140)
        self.assertGreater(events[0]['rejected'],0)
        self.assertTrue((raw['setpiece'][140:300]==0).all())
        self.assertTrue((raw['preparation'][140:300]==0).all())
        self.assertTrue((raw['after_restart'][140:215]==1).all())

    def test_overlap_keeps_new_unrestarted_event_and_no_candidate_keeps_legacy(self):
        d=self.data();self.play(d,70)
        tags=[dict(code='SET PIECES',start_frame=0,end_frame=250,tags={'Type':'Goal Kick'}),
              dict(code='SET PIECES',start_frame=160,end_frame=350,tags={'Type':'Free Kick'})]
        raw,events=self.features(d,tags,detector='contact',retirement='tags')
        self.assertTrue((raw['setpiece'][160:350]==1).all())
        self.assertTrue((raw['goal_tag'][70:]==0).all())
        self.assertIsNone(events[1]['restart'])
        d=self.data();raw,_=self.features(d,tags,detector='contact',retirement='tags')
        old,_=self.features(d,tags,detector='best',retirement='legacy')
        for k in raw:np.testing.assert_array_equal(raw[k],old[k])

    def test_out_of_field_cannot_confirm_restart(self):
        d=self.data();self.play(d,70);d['ball'][:,1]=80
        self.assertFalse(confirmation(d,70,0,250,'play',4)[0])

    def test_two_stationary_clusters_across_missing_gap_are_not_a_pass(self):
        d=self.data();d['speed'][70:140]=8;d['contact'][65:75]=True
        d['seen'][85:115]=False;d['ball'][115:,0]=55
        self.assertFalse(confirmation(d,70,0,250,'contact',4)[0])


if __name__=='__main__':unittest.main()
