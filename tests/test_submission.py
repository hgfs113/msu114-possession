import unittest
import numpy as np
from possession.__main__ import submission_labels


class SubmissionTests(unittest.TestCase):
    def test_published_pre_kickoff_block_is_dead(self):
        labels=np.full(199319,'X',dtype='<U1')
        labels[11185:90652]='H';labels[113742:190198]='A'
        result=submission_labels(labels)
        self.assertTrue((result[110808:113742]=='D').all())
        self.assertEqual(result[110807],'X');self.assertEqual(result[113742],'A')
        self.assertEqual(result[190198],'X')
        np.testing.assert_array_equal(result[11185:90652],labels[11185:90652])
        np.testing.assert_array_equal(result[113742:190198],labels[113742:190198])


if __name__=='__main__':unittest.main()
