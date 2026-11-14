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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Directory containing tactical/ and broadcast/')
    parser.add_argument('--output',type=Path,required=True,help='Private output CSV, not committed to Git')
    parser.add_argument('--cache-dir',type=Path,default=Path('.cache'))
    args=parser.parse_args()
    os.environ['MTAH_DATA_DIR']=str(args.data_dir.resolve())
    os.environ['POSSESSION_CACHE_DIR']=str(args.cache_dir.resolve())
    from games import asset_paths
    missing=[p for source in ('tactical','broadcast') for name,p in asset_paths(source=source).items()
             if name!='video' and not Path(p).is_file()]
    if missing:parser.error('Missing organiser assets: '+', '.join(missing))
    started=time.monotonic()
    import numpy as np
    from possession.activity.model import predict,freeze_ownership,compose
    from possession.base import write_prediction,N
    config=json.loads(Path(__file__).with_name('activity').joinpath('interval_config.json').read_text())
    activity=predict(config)
    _,owner=freeze_ownership()
    labels=submission_labels(np.array(list('HADX'))[compose(activity,owner)])
    assert len(labels)==N and set(labels)<=set('HADX')
    write_prediction(args.output,labels)
    print(json.dumps(dict(output=str(args.output.resolve()),rows=len(labels),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),seconds=round(time.monotonic()-started,2))))


if __name__=='__main__':main()
