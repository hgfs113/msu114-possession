import unittest
import numpy as np
from possession.goalkeeper.phase_features import measured_velocity
from possession.goalkeeper.phases import phase_evidence,decode_phases
from possession.whistle.fusion import decode


class KeeperPhaseTests(unittest.TestCase):
    def features(self,n=70):
        f={key+'0':np.full((n,2),value,dtype=float) for key,value in [('distance',30),('keeper_seen',1),('keeper_box',1),('ball_box',1),
            ('keeper_speed',0),('radial',0),('relative_speed',20),('tca',100),('miss_distance',100),('goal_area',0)]}
        f.update(seen0=np.ones(n),ball_speed0=np.ones(n)*8,other0=np.ones(n)*40,outside0=np.zeros(n),goal_tag0=np.zeros(n),keeper_xy0=np.zeros((n,2,2)))
        f['distance0'][5:10,0]=7;f['radial0'][5:10,0]=-5
        f['distance0'][10:35,0]=1;f['ball_speed0'][10:35]=0;f['radial0'][10:35,0]=0
        f['seen0'][13:35]=0;f['distance0'][13:35]=1000
        f['distance0'][35:37,0]=[5,8];f['radial0'][35:37,0]=8
        return f

    def evidence(self,f,c=None,audio=None):
        n=len(f['seen0']);return phase_evidence(f,0,np.ones(n,bool),audio or dict(role=np.zeros(n)),c or dict(anchor='visible',gate='context',require_release=True))

    def test_receive_hold_release_chain(self):
        p=self.evidence(self.features())
        self.assertTrue(p['receive'][10,0]>0);self.assertTrue(p['hold'][25,0]>0);self.assertTrue(p['release'][35,0]>0)
        self.assertEqual(p['hold'][35,0],0);self.assertEqual(p['episodes'][0]['end_reason'],'release')

    def test_static_ball_is_not_receive_and_missing_alone_is_not_receive(self):
        f=self.features();f['distance0'][:]=1;f['ball_speed0'][:]=0;f['radial0'][:]=0
        self.assertFalse(self.evidence(f)['receive'].any())
        f=self.features();f['seen0'][:]=0
        self.assertFalse(self.evidence(f)['receive'].any())

    def test_closed_episode_requires_release(self):
        f=self.features();f['seen0'][35:]=0
        self.assertFalse(self.evidence(f)['hold'].any())
        p=self.evidence(f,dict(anchor='visible',gate='context',require_release=False))
        self.assertTrue(p['hold'].any())

    def test_goal_kick_and_outside_context_reject_false_catch(self):
        f=self.features();f['goal_area0'][:,0]=1;f['goal_tag0'][:]=1
        self.assertFalse(self.evidence(f)['hold'].any())
        f=self.features();f['outside0'][8]=1
        self.assertFalse(self.evidence(f)['hold'].any())

    def test_new_stop_interrupts_open_hold_and_no_state_carries_to_fresh_half(self):
        f=self.features();audio=dict(role=np.zeros(70));audio['role'][20:23]=1
        p=self.evidence(f,dict(anchor='visible',gate='context',require_release=False),audio)
        self.assertEqual(p['hold'][20,0],0)
        sliced={k:v[15:] for k,v in f.items()}
        self.assertFalse(self.evidence(sliced)['hold'].any())

    def test_unknown_or_jumping_player_motion_is_not_static(self):
        p=np.array([[0.,0.],[np.nan,np.nan],[1,0],[100,0]])
        self.assertTrue(np.isnan(measured_velocity(p,np.arange(4)*.2,12)).all())

    def test_decoder_has_no_keeper_without_reception_and_preserves_control(self):
        rng=np.random.default_rng(6);score=rng.normal(0,5,60);p={k:np.zeros((60,2)) for k in ('receive','hold','release','setup')}
        np.testing.assert_array_equal(decode_phases(score,p),decode(score,15,15))
        p['hold'][10:40,0]=1
        self.assertFalse(np.isin(decode_phases(score,p),[2,3,4]).any())

    def test_setup_exports_dead_and_receive_can_restore_keeper_control(self):
        p=self.evidence(self.features());score=np.ones(70)*2
        path=decode_phases(score,p,12)
        self.assertTrue(np.all(path[16:30]==3));self.assertFalse((path[45:]==3).any())
        p={k:np.zeros((70,2)) for k in ('receive','hold','release','setup')};p['setup'][10:40,0]=1
        path=decode_phases(score,p,12)
        self.assertTrue(np.all(np.isin(path[10:40],[1,5])))


if __name__=='__main__':unittest.main()
