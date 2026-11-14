import unittest
import numpy as np
from possession.kinematics import kinematic_filter,sustained_contact,decode_general
from possession.preprocess import PITCH

class ContactChecks(unittest.TestCase):
    def test_rts_constant_velocity_and_gap(self):
        n=100;t=np.arange(n)*.04;z=np.c_[10+2*t,20+t]/PITCH
        z[40:44]=np.nan
        pos,vel,var,obs=kinematic_filter(z,noise=.2,process=2,gap=.4,rts=True,halves=((0,n),))
        np.testing.assert_allclose(pos[10:90,0,0],10+2*t[10:90],atol=.04)
        np.testing.assert_allclose(vel[10:90,0,0],2,atol=.1)
        self.assertTrue(np.isfinite(pos[40:44]).all());self.assertFalse(obs[40:44].any());self.assertTrue((var>=0).all())
    def test_rts_reset_does_not_cross_teleport_or_half(self):
        z=np.tile([.1,.5],(100,1));z[50:]=[.9,.5]
        p,v,_,_=kinematic_filter(z,rts=True,halves=((0,100),))
        self.assertLess(p[49,0,0],11);self.assertGreater(p[50,0,0],90)
        p,v,_,_=kinematic_filter(z,rts=True,halves=((0,45),(55,100)))
        self.assertTrue(np.isnan(p[45:55]).all())
    def test_rts_reduces_noise_and_covariance(self):
        rng=np.random.default_rng(42);n=300;t=np.arange(n)*.04
        truth=np.c_[20+2*t,30+.5*t];z=(truth+rng.normal(0,.25,truth.shape))/PITCH
        pf,_,vf,_=kinematic_filter(z,noise=.25,process=.5,rts=False,halves=((0,n),))
        ps,_,vs,_=kinematic_filter(z,noise=.25,process=.5,rts=True,halves=((0,n),))
        self.assertLess(np.mean((ps[20:-20,0]-truth[20:-20])**2),np.mean((pf[20:-20,0]-truth[20:-20])**2))
        self.assertTrue((vs<=vf+1e-6).all())
    def test_contacts_require_observation(self):
        # Full-length synthetic arrays avoid reliance on match assets.
        n=199319;distance=np.full((n,2),20.);distance[11185:11195,0]=.5
        actual=np.ones((n,2),bool);actual[11188:11190]=False
        f=dict(distance=distance,actual=actual,slots=np.zeros((n,2),int),relv=np.zeros((n,2)),uncertainty=np.zeros((n,2)))
        contact=sustained_contact(f,dict(radius=1,contact_frames=3));self.assertTrue(contact[11185:11188,0].all());self.assertFalse(contact[11188:11190].any());self.assertFalse(contact[:,1].any())
if __name__=='__main__':unittest.main()
