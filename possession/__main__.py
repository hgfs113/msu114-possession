"""Reproduce the fixed team-possession model from local organiser assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def submission_labels(labels):
    """Use the published evaluation blocks, including the pre-kickoff DEAD span."""
    import numpy as np
    from games import GAME, half_video_blocks
    from possession.preprocess import HALVES
    result=np.full(len(labels),'X',dtype='<U1')
    for (_,first,last),(kickoff,end) in zip(half_video_blocks(GAME),HALVES):
        result[first:kickoff]='D'
        result[kickoff:min(end,last+1)]=labels[kickoff:min(end,last+1)]
    return result


def main():
    from possession.release import main as run
    run()


if __name__=='__main__':main()
