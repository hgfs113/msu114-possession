"""Reproduce fixed tracking/audio possession inference from local assets."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--tactical-video',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['MTAH_DATA_DIR']=str(a.data_dir.resolve())
    os.environ['POSSESSION_CACHE_DIR']=str(a.cache_dir.resolve())
    from games import asset_paths
    for source in ('tactical','broadcast'):
        for name,path in asset_paths(source=source).items():
            if name!='video' and not Path(path).is_file():p.error('Missing organiser asset: '+str(path))
    if not a.tactical_video.is_file():p.error('Tactical video is missing')
    from possession.runtime import cache_directory
    from possession.whistle.detect import extract,sha
    from possession.whistle.referee_v3 import extract_features,detect
    from possession.whistle.activity_v31 import prepare_pack
    from possession.goalkeeper.phase_features import prepare as prepare_keeper
    from possession.restarts import prepare as prepare_restarts,RestartModel
    from possession.base import write_prediction,N
    started=time.monotonic();cache=cache_directory();cfgdir=Path(__file__).parent
    stage=cache/'release';stage.mkdir(parents=True,exist_ok=True)
    # Generated files remain in the private cache, never in the source checkout.
    physical=stage/'physical.npz';keeper=stage/'keeper.npz';restarts=stage/'restart.npz'
    audio=stage/'tactical_audio_16k.wav';detections=stage/'detections.json'
    print('Preparing tracking, camera alignment and physical evidence',flush=True)
    if not physical.exists():prepare_pack(physical)
    print('Extracting tactical audio and scanning whistle proposals',flush=True)
    extract(a.tactical_video,audio)
    features,identity=extract_features(audio,stage/'audio_features.npz')
    model=json.loads((cfgdir/'whistle/fixed_config.json').read_text())
    events=detect(features,model['config'])
    detections.write_text(json.dumps(dict(recording_source='tactical',events=events,
        identity=identity,config=model['config'],model_sha256=sha(cfgdir/'whistle/fixed_config.json')),indent=2)+'\n')
    print('Preparing reception/release and restart features',flush=True)
    if not keeper.exists():prepare_keeper(cache/'tracking_raw.npz',keeper,a.data_dir)
    if not restarts.exists():prepare_restarts(restarts,cache,a.data_dir)
    c=json.loads((cfgdir/'selected_config.json').read_text());base=c.pop('base_config')
    prediction=RestartModel(restarts,base,physical,keeper,detections)
    labels,details=prediction.predict(c)
    if len(labels)!=N or not set(labels)<=set('HADX'):raise ValueError('Invalid generated labels')
    a.output.parent.mkdir(parents=True,exist_ok=True);write_prediction(a.output,labels)
    report=dict(output=str(a.output.resolve()),sha256=sha(a.output),frames=N,
        config=c,base_config=base,whistles=len(events),restart_windows=len(details),
        feature_reconstruction_errors=prediction.baseline_error,
        assets={str(path):sha(path) for path in (physical,keeper,restarts,audio,detections)},
        seconds=round(time.monotonic()-started,2))
    a.output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('output','sha256','frames','whistles','seconds')}),flush=True)


if __name__=='__main__':main()
