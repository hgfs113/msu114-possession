"""Synthetic behavioural tests, no match data or reference needed."""
import unittest
import numpy as np
from possession.preprocess import fix_goalkeepers,kalman
from possession.base import viterbi
class Behaviour(unittest.TestCase):
    def test_keeper_sides_and_missing(self):
        x=np.full((4,46,2),np.nan);x[:,23]=[.9,.2];x[:,45]=[.1,.7]
        out,flag=fix_goalkeepers(x,((0,2),(2,4)))
        np.testing.assert_allclose(out[:2,23,0],.1);np.testing.assert_allclose(out[:2,45,0],.9)
        np.testing.assert_allclose(out[2:],x[2:]);np.testing.assert_allclose(out[:,:,1],x[:,:,1]);self.assertTrue(np.isnan(out[:,0]).all())
        again,_=fix_goalkeepers(out,((0,2),(2,4)));np.testing.assert_allclose(again,out)
    def test_viterbi_global_and_duration(self):
        e=np.tile([2.,-2.,-2.],(30,1));e[10]=[-2,2,-2]
        out=viterbi(e,8,8);self.assertTrue((out==0).all())
        e[15:]=[-2,2,-2];out=viterbi(e,3,3);self.assertTrue((out[:15]==0).all());self.assertTrue((out[15:]==1).all())
        out=viterbi(e,3,3,3);cuts=np.r_[0,np.flatnonzero(np.diff(out))+1,len(out)];self.assertTrue((np.diff(cuts)[:-1]>=3).all())
    def test_kalman_missing_and_half_reset(self):
        x=np.full((199319,2),np.nan);x[11185:11195]=[.2,.5];x[113742:113752]=[.8,.5]
        y=kalman(x,ball=True,max_gap=.12)
        self.assertTrue(np.isnan(y[11198]).all());np.testing.assert_allclose(y[113742],[.8,.5]);self.assertTrue(np.isnan(y[110000]).all())
if __name__=='__main__':unittest.main()
