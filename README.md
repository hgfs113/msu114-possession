# msu114 - Football possession inference

Offline team possession and dead-ball inference from tracking, tactical tags,
camera geometry and tactical-video audio. The fixed pipeline combines adaptive
Kalman/RTS kinematics, temporal decoding, whistle context, goalkeeper reception
and release, and a detector that closes set-piece preparation after a restart.

## Installation

Python 3.11, NumPy and pandas are required. Install the pinned dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` and put both on PATH. The verified version is 8.1.2. Audio extraction uses the
first audio stream of the tactical MP4, converted to mono PCM16 at 16 kHz.
No GPU, cloud service or downloaded neural-network weights are required.

## Local inputs

Keep organiser assets outside the repository:

```text
/path/to/match-data/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
    BAR-ATH_tactical.mp4
  broadcast/
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
```

The MP4 passed below must use the tactical frame clock and zero audio/video
stream start timestamps. Broadcast inputs provide geometric context; the
broadcast video is not required by this configuration.

## Reproduce the CSV

From the repository root:

```bash
.venv/bin/python -m possession \
  --data-dir /path/to/match-data \
  --tactical-video /path/to/match-data/tactical/BAR-ATH_tactical.mp4 \
  --cache-dir /path/to/private-cache \
  --output /path/to/private-output/msu114.csv

.venv/bin/python evaluation/validate_submission.py /path/to/private-output/msu114.csv
```

The output is UTF-8 `frame,label`, exactly 199,319 data rows indexed 0..199318,
at 25 FPS on the tactical timeline: H = Barcelona, A = Athletic Club,
D = dead ball, X = outside the exported evaluation blocks.

The command prepares all features and audio detections locally, then prints
elapsed time and the CSV SHA-256. A private companion manifest records the
configuration and generated feature hashes. Caches and outputs must not be
committed. For a fresh reproduction use an empty cache directory.

Expected CSV SHA-256 for the supplied match assets:
`0df2522f96caf1529801c14e2dc8711959247daf500cb3e4e9df17d9e4c4e6a1`.

Each half is decoded independently with the same configuration. Added time is
included; halftime is excluded. The published pre-kickoff part of the second
block is exported as D. Centred windows, RTS, restart confirmation and Viterbi
use future observations, so this is an offline system.

## Implementation

- `possession/release.py`: complete inference command and private-cache preparation.
- `possession/kinematics.py`, `ownership.py`, `cameras.py`: movement, control and camera context.
- `possession/activity/`: physical activity evidence and logistic model.
- `possession/whistle/`: paired-tone detection and context-dependent acoustic evidence.
- `possession/goalkeeper/`: observed reception, holding, release and standard preparation.
- `possession/restarts.py`: restart confirmation and retirement of stale preparation tags.
- `possession/selected_config.json`: fixed final configuration.

[Technical document (PDF)](msu114.pdf) describes the runtime method and its
limitations. [Markdown version](TECHNICAL_REPORT.md).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Synthetic tests cover missing data, filtering resets, goalkeeper-side correction,
phase transitions, audio context, restart confirmation, overlapping tags and
CSV frame conventions. Format validation does not measure predictive quality.

## Attribution and data

This implementation builds on the MIT-licensed Metrica Sports starter repository:
https://github.com/metrica-sports/more-than-a-hack-2027 .
The upstream loaders, evaluation code and copyright notice are retained.
Match assets, video, audio, tracking exports, predictions and caches are not
included. Their separate data terms are not changed by the source-code licence.
