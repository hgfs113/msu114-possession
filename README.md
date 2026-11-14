# msu114 · Football possession inference

Offline inference of team possession and dead-ball intervals from automatic football tracking. Team: **msu114**.

The fixed pipeline combines adaptive Kalman/RTS kinematics, observed close-control evidence, automatic tactical tags, camera reliability, conservative broadcast correspondences and a separate binary activity model. It writes a `frame,label` CSV on the tactical-video timeline.

## Setup

Python **3.11** is the verified environment. From this repository:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt`. No GPU, video decoder, pretrained weights or network connection is needed for inference.

## Local input files

Obtain the assets through the organisers' registration process. Keep them outside this repository. Expected structure:

```text
/path/to/match-data/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
  broadcast/
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
```

MP4 files are not required by this inference pipeline. All coordinates, tags and homographies are read locally. Match assets, cached arrays, predictions and video are excluded from Git.

## Reproduce the submission

```bash
.venv/bin/python -m possession \
  --data-dir /path/to/match-data \
  --cache-dir /path/to/private-cache \
  --output /path/to/private-output/msu114.csv

.venv/bin/python evaluation/validate_submission.py /path/to/private-output/msu114.csv
```

The CSV contains exactly **199,319 data rows**, indexed `0..199318` at 25 FPS:

- `H`: Barcelona possession while the ball is in play.
- `A`: Athletic Club possession while the ball is in play.
- `D`: dead ball.
- `X`: outside the published evaluation blocks.

The published pre-kickoff portion of the second evaluation block is exported as `D`. Each half is decoded independently. The algorithm is offline: centred windows, RTS smoothing and Viterbi use future observations.

The first run builds local features and cross-camera correspondences; subsequent runs reuse a private cache. The cache namespace incorporates source paths, sizes, modification times, implementation and configuration. Use a new cache directory to verify a completely fresh run. The command prints the output SHA-256 and elapsed time.

## Tests

Synthetic tests do not require the match assets:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

They cover missing observations, RTS resets, goalkeeper-side correction, temporal persistence, physical pseudo-label learning, activity composition and the published frame contract. Format validation is separate from predictive accuracy.

## Implementation

- `possession/tracking.py`: ATD loading and local geometric synchronization.
- `possession/kinematics.py`: adaptive Kalman/RTS and observed close controls.
- `possession/ownership.py`, `context.py`, `cameras.py`: team evidence and temporal decoding.
- `possession/activity/`: physical pseudo-labels, balanced logistic regression and binary HMM.
- `possession/__main__.py`: reproducible export.
- [Technical report](TECHNICAL_REPORT.md): method, input requirements and limitations.

## Attribution

This implementation builds on the MIT-licensed [Metrica Sports starter repository](https://github.com/metrica-sports/more-than-a-hack-2027). `games.py`, the two loaders in `starter/`, and the validation/metric code in `evaluation/` originate from that repository. The upstream copyright and license are retained in [LICENSE](LICENSE).

No match assets or ground-truth files are included. Their separate terms are not changed by this repository's code license.
