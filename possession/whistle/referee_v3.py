"""Audio-only paired-tone whistle proposals over the complete recording."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from .detect import intervals, sha

VERSION = 'full-scan-paired-tone-v3'
SR, WINDOW, HOP = 16000, 2048, 320
# Fixed paired-tone frequency profile.
PROFILE = dict(carrier_hz=3765.625, gap_hz=460.9375,
               carrier_tolerance_hz=180., gap_tolerance_hz=80.)


def frame_features(power):
    """Return both simultaneous tones; do not require the lower tone to be louder."""
    freq = np.fft.rfftfreq(WINDOW, 1/SR)
    lower = np.flatnonzero(abs(freq-PROFILE['carrier_hz']) <= PROFILE['carrier_tolerance_hz'])
    gaps = np.flatnonzero(abs(freq-PROFILE['gap_hz']) <= PROFILE['gap_tolerance_hz'])
    upper = lower[:, None]+gaps[None, :]
    # Local background excludes the carrier and its Hann main lobe.
    shifts = np.r_[np.arange(-24, -7), np.arange(8, 25)]
    bins = np.arange(lower.min(), upper.max()+1)
    background = np.median(power[:, bins[:, None]+shifts], axis=2)
    prominence = 10*np.log10(power[:, bins]/np.maximum(background, 1e-16))
    lo = prominence[:, lower-bins[0]]
    hi = prominence[:, upper-bins[0]]
    # A pair cannot be rescued by one very loud single tone.
    score = np.minimum(lo[:, :, None], hi)
    score -= 3*(abs(freq[lower]-PROFILE['carrier_hz'])/PROFILE['carrier_tolerance_hz'])[None, :, None]
    score -= 3*(abs(freq[gaps]-PROFILE['gap_hz'])/PROFILE['gap_tolerance_hz'])[None, None, :]
    flat = score.reshape(len(power), -1).argmax(1)
    row = np.arange(len(power)); a, b = np.unravel_index(flat, score.shape[1:])
    lower_index, upper_index = lower[a], upper[a, b]
    return dict(score_db=score[row, a, b], carrier_hz=freq[lower_index],
                gap_hz=freq[upper_index]-freq[lower_index],
                tone_power=power[row, lower_index]+power[row, upper_index])


def extract_features(wav, cache=None):
    identity = dict(wav_sha256=sha(wav), code_sha256=sha(__file__), profile=PROFILE)
    if cache:
        cache = Path(cache); key = cache.with_suffix('.json')
        if cache.exists() and key.exists() and json.loads(key.read_text()) == identity:
            with np.load(cache) as data: return dict(data), identity
    with wave.open(str(wav), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (SR, 1, 2):
            raise ValueError('Expected mono PCM16 at 16 kHz')
        count = max(0, 1+(audio.getnframes()-WINDOW)//HOP)
        features = {k: np.empty(count, 'f4') for k in
                    ('time', 'score_db', 'carrier_hz', 'gap_hz', 'tone_power', 'rms_db')}
        window = np.hanning(WINDOW)
        for a in range(0, count, 256):
            b = min(count, a+256); audio.setpos(a*HOP)
            signal = np.frombuffer(audio.readframes((b-a-1)*HOP+WINDOW), '<i2').astype(float)/32768
            frames = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)[::HOP][:b-a]
            power = abs(np.fft.rfft(frames*window, axis=1))**2+1e-16
            values = frame_features(power)
            values.update(time=(np.arange(a, b)*HOP+WINDOW/2)/SR,
                          rms_db=20*np.log10(np.sqrt((frames**2).mean(1))+1e-10))
            for k in features: features[k][a:b] = values[k]
        features['audio_duration'] = np.array(audio.getnframes()/SR)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **features); key.write_text(json.dumps(identity, indent=2)+'\n')
    return features, identity


def detect(features, config):
    support = (features['score_db'] >= config['threshold_db']) & (features['rms_db'] >= -65)
    joined = support.copy()
    for a, b in intervals(~support):
        if a > 0 and b < len(support) and (b-a)*HOP/SR <= .10: joined[a:b] = True
    events = []
    for a, b in intervals(joined):
        active = np.flatnonzero(support[a:b])+a
        if len(active)*HOP/SR < config['support_seconds']: continue
        if (b-a)*HOP/SR > 4.: continue  # prolonged crowd tone remains a rejected candidate
        carrier = features['carrier_hz'][active]
        if np.median(abs(carrier-np.median(carrier))) > 70: continue
        peak = int(active[np.argmax(features['score_db'][active])])
        start = max(0., float(features['time'][a])-WINDOW/2/SR)
        end = min(float(features['audio_duration']), float(features['time'][b-1])+WINDOW/2/SR)
        events.append(dict(id=f'v3_{len(events)+1:04d}', start_seconds=round(start, 4),
            end_seconds=round(end, 4), peak_seconds=round(float(features['time'][peak]), 4),
            score_db=round(float(np.median(features['score_db'][active])), 3),
            carrier_hz=round(float(np.median(carrier)), 2),
            gap_hz=round(float(np.median(features['gap_hz'][active])), 2),
            support_seconds=round(len(active)*HOP/SR, 3)))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--recording', choices=('tactical','broadcast'), default='tactical')
    args = parser.parse_args(); model = json.loads(args.model.read_text())
    if model['version'] != VERSION or model['profile'] != PROFILE: raise ValueError('Incompatible model')
    features, identity = extract_features(args.wav, args.cache)
    result = dict(version=VERSION, recording_source=args.recording, identity=identity,
                  model_sha256=sha(args.model), duration_seconds=float(features['audio_duration']),
                  config=model['config'], events=detect(features, model['config']),
                  inference_inputs=['audio waveform', 'frozen model'], uses_op_at_inference=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(events=len(result['events']), output=str(args.output))))


if __name__ == '__main__': main()
