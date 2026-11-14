import unittest
import numpy as np
from possession.activity.model import binary_decode,train_logistic,sigmoid,physical_seeds,compose
from possession.activity.features import reduce_blocks,NAMES



class ActivityTests(unittest.TestCase):
    def test_hmm_preserves_sustained_stop_but_rejects_single_spike(self):
        x=np.full(100,-2.);x[10]=4;x[40:60]=4
        p=binary_decode(x,8,8)
        self.assertEqual(p[10],0);self.assertTrue((p[40:60]==1).all());self.assertEqual(int(p.sum()),20)

    def test_reduce_never_crosses_half_boundaries(self):
        x=np.array([100,1,1,1,100,100,2,2,2,100])
        a,b=reduce_blocks(x,halves=[(1,4),(6,9)],step=2)
        np.testing.assert_array_equal(a,[1,1]);np.testing.assert_array_equal(b,[2,2])

    def test_logistic_can_learn_physical_labels(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(500,3));y=(x[:,0]+x[:,1]>.3).astype(float)
        p=train_logistic(x,y,.01);prob=sigmoid(((x-p['mean'])/p['scale'])@p['weight']+p['intercept'])
        self.assertGreater(((prob>.5)==y).mean(),.97)


    def test_composition_preserves_owner_and_outside_halves(self):
        out=compose(np.array(list('XLLDDX')),np.array([3,0,1,0,1,3]))
        np.testing.assert_array_equal(out,[3,0,1,2,2,3])


if __name__=='__main__':unittest.main()
