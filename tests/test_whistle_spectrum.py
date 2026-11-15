import unittest
import numpy as np

from possession.whistle.referee_v3 import frame_features, WINDOW, SR


class WhistleSpectrumTests(unittest.TestCase):
    def score(self,low,high):
        time=np.arange(WINDOW)/SR
        signal=low*np.sin(2*np.pi*3765.625*time)+high*np.sin(2*np.pi*4226.5625*time)
        signal+=np.random.default_rng(1).normal(0,.05,WINDOW)
        power=abs(np.fft.rfft(signal*np.hanning(WINDOW)))**2+1e-16
        return frame_features(power[None,:])

    def test_two_tones_with_either_component_dominant(self):
        for low,high in ((1,.4),(.4,1)):
            f=self.score(low,high)
            self.assertGreater(f['score_db'][0],25)
            self.assertAlmostEqual(f['gap_hz'][0],460.9375)

    def test_single_tone_and_noise_are_not_a_confirmed_pair(self):
        for low,high in ((1,0),(0,1),(0,0)):
            self.assertLess(self.score(low,high)['score_db'][0],18)


if __name__=='__main__':unittest.main()
