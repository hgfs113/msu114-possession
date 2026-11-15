"""Local STFT whistle proposals. No possession labels or events are read."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import numpy as np

VERSION = 'spectral-whistle-v1'
SAMPLE_RATE = 16000
WINDOW = 1024
HOP = 160
FPS = 25
CONFIG = dict(band_hz=[2800, 4800], primary_band_hz=[3300, 4200],
              min_peak_db=16, min_concentration=.30, min_band_ratio=.10,
              min_rms_db=-65, min_duration=.12, gap_seconds=.09,
              max_frequency_mad_hz=220)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(video, output):
    video, output = Path(video).resolve(), Path(output)
    metadata = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration,start_time:stream=index,codec_type,codec_name,sample_rate,channels,start_time,duration',
        '-of', 'json', str(video)], text=True))
    audio = next((s for s in metadata['streams'] if s['codec_type'] == 'audio'), None)
    picture = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), None)
    if audio is None or picture is None:
        raise ValueError('Both audio and video streams are required')
    video_start = float(picture.get('start_time', 0))
    audio_start = float(audio.get('start_time', 0))
    # Explicitly refuse unverified clock origins rather than silently shifting detections.
    if abs(video_start) > .001 or abs(audio_start) > .001:
        raise ValueError('Nonzero stream timestamps require explicit alignment before extraction')
    signature = dict(source_name=video.name, source_bytes=video.stat().st_size,
                     source_mtime_ns=video.stat().st_mtime_ns, streams=metadata,
                     sample_rate=SAMPLE_RATE, channels=1, pcm_bits=16)
    manifest = output.with_suffix('.source.json')
    if not (output.is_file() and manifest.is_file() and json.loads(manifest.read_text()) == signature):
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.stem + '.partial.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(video),
                        '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                        '-c:a', 'pcm_s16le', str(temporary)], check=True)
        temporary.replace(output)
        manifest.write_text(json.dumps(signature, indent=2))
    return signature


def spectral_features(wav_path):
    """Bounded-memory FFT batches; times denote analysis-window centres."""
    with wave.open(str(wav_path), 'rb') as sound:
        assert sound.getnchannels() == 1 and sound.getsampwidth() == 2
        sr = sound.getframerate()
        if sr != SAMPLE_RATE:
            raise ValueError('Expected mono 16 kHz PCM16 WAV')
        count = max(0, 1 + (sound.getnframes() - WINDOW) // HOP)
        names = ('time', 'peak_db', 'concentration', 'band_ratio', 'rms_db', 'frequency', 'peak_power')
        result = {name: np.empty(count, 'f4') for name in names}
        frequency = np.fft.rfftfreq(WINDOW, 1 / sr)
        band = np.flatnonzero((frequency >= CONFIG['band_hz'][0]) & (frequency <= CONFIG['band_hz'][1]))
        wide = (frequency >= 500) & (frequency <= 7500)
        window = np.hanning(WINDOW).astype('f4')
        for first in range(0, count, 2048):
            last = min(count, first + 2048)
            sound.setpos(first * HOP)
            signal = np.frombuffer(sound.readframes((last-first-1)*HOP+WINDOW), '<i2').astype('f4') / 32768
            frames = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)[::HOP][:last-first]
            spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 + 1e-16
            selected = spectrum[:, band]
            index = selected.argmax(1)
            peak = selected[np.arange(len(index)), index]
            narrow = np.zeros(len(index))
            for offset in (-2, -1, 0, 1, 2):
                narrow += selected[np.arange(len(index)), np.clip(index+offset, 0, len(band)-1)]
            values = dict(time=(np.arange(first, last)*HOP+WINDOW/2)/sr,
                          peak_db=10*np.log10(peak / np.median(selected, axis=1)),
                          concentration=np.minimum(1, narrow/selected.sum(1)),
                          band_ratio=selected.sum(1)/spectrum[:, wide].sum(1),
                          rms_db=20*np.log10(np.sqrt((frames**2).mean(1))+1e-10),
                          frequency=frequency[band[index]], peak_power=peak)
            for name in names:
                result[name][first:last] = values[name]
        result['audio_duration'] = np.array(sound.getnframes()/sr)
    return result


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype('i1'))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def proposals(features, config=CONFIG):
    f = features
    selected = ((f['peak_db'] >= config['min_peak_db']) &
                (f['concentration'] >= config['min_concentration']) &
                (f['band_ratio'] >= config['min_band_ratio']) &
                (f['rms_db'] >= config['min_rms_db']))
    joined = selected.copy()
    for a, b in intervals(~selected):
        if a > 0 and b < len(selected) and (b-a)*HOP/SAMPLE_RATE <= config['gap_seconds']:
            joined[a:b] = True
    events = []
    for a, b in intervals(joined):
        support = np.flatnonzero(selected[a:b]) + a
        duration = (b-a)*HOP/SAMPLE_RATE
        if len(support)*HOP/SAMPLE_RATE < config['min_duration']:
            continue
        centre = float(np.median(f['frequency'][support]))
        mad = float(np.median(np.abs(f['frequency'][support]-centre)))
        if mad > config['max_frequency_mad_hz']:
            continue
        peak_db = float(np.percentile(f['peak_db'][support], 75))
        ratio = float(np.percentile(f['band_ratio'][support], 75))
        concentration = float(np.percentile(f['concentration'][support], 75))
        score = float(.45*np.clip((peak_db-12)/20, 0, 1) +
                      .30*np.clip((ratio-.05)/.55, 0, 1) +
                      .25*np.clip((concentration-.2)/.65, 0, 1))
        # Long continuous tones and very short fragments remain reviewable but rank lower.
        score *= min(1., duration/.22) * min(1., 3/max(duration, 1e-6))
        start = max(0., float(f['time'][a])-WINDOW/2/SAMPLE_RATE)
        end = min(float(f['audio_duration']), float(f['time'][b-1])+WINDOW/2/SAMPLE_RATE)
        peak_index = int(support[np.argmax(f['peak_power'][support])])
        events.append(dict(id=f'w{len(events)+1:04d}', start_seconds=round(start, 4),
                           end_seconds=round(end, 4), peak_seconds=round(float(f['time'][peak_index]), 4),
                           start_frame=int(round(start*FPS)), end_frame=int(round(end*FPS)),
                           peak_frame=int(round(float(f['time'][peak_index])*FPS)),
                           score=round(score, 4), frequency_hz=round(centre, 1),
                           frequency_mad_hz=round(mad, 1), peak_db=round(peak_db, 2),
                           band_ratio=round(ratio, 4), concentration=round(concentration, 4),
                           duration_seconds=round(end-start, 4), label='unreviewed'))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/whistles'))
    parser.add_argument('--source', choices=('tactical', 'broadcast'), default='tactical',
                        help='Recording clock used by the detections; broadcast is not tactical time')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f'{args.source}_audio_16k.wav'
    source = extract(args.video, wav)
    feature_file = out / 'spectral_features.npz'
    key = sha(wav) + sha(Path(__file__))
    keyfile = out / 'spectral_features.key'
    if feature_file.exists() and keyfile.exists() and keyfile.read_text() == key:
        with np.load(feature_file) as loaded:
            features = dict(loaded)
    else:
        features = spectral_features(wav)
        np.savez_compressed(feature_file, **features)
        keyfile.write_text(key)
    events = proposals(features)
    report = dict(version=VERSION, config=CONFIG, source=source, recording_source=args.source,
                  alignment=dict(audio_video_origin_seconds=0, fps=FPS,
                                 note='PTS aligned; acoustic propagation and capture delay are not corrected'),
                  score_semantics='spectral evidence, not calibrated probability or confirmed referee identity',
                  events=events, duration_seconds=float(features['audio_duration']),
                  wav_sha256=key[:64], detector_sha256=sha(Path(__file__)))
    (out / 'detections.json').write_text(json.dumps(report, indent=2) + '\n')
    if events:
        with (out / 'detections.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(events[0]))
            writer.writeheader(); writer.writerows(events)
    print(json.dumps(dict(events=len(events), high_score=int(sum(e['score'] >= .7 for e in events)),
                          duration_seconds=report['duration_seconds'], output=str(out))))


if __name__ == '__main__':
    main()
