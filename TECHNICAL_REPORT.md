# msu114 - Tracking and audio possession inference

## 1. Objective and runtime inputs

The offline system assigns H (Barcelona), A (Athletic Club), D (dead ball) or X
(outside the evaluation blocks) to each of 199,319 tactical-video frames at
25 FPS. Runtime inputs are automatic tracking and metadata from both cameras,
their homographies, tactical Smart Tagging and the tactical MP4 audio track.
No video-pixel model or player-name database is used.

## 2. Geometry and team ownership

Coordinates are converted to metres on a 105.3 x 68 m pitch. A fixed goalkeeper
role hypothesis mirrors X for slots 24 and 46 when they appear on the opposite
defending half; sides reverse at halftime. It preserves Y and missing values.
Slots are not treated as persistent player identities.

Adaptive constant-velocity Kalman filtering and RTS smoothing estimate positions
and velocities. Gap extrapolation is bounded and halftime resets the state.
Observed close control requires distance at most 1 m, relative speed at most
3 m/s, an advantage of at least 0.5 m over the opponent and three observed
consecutive frames. Predicted coordinates alone cannot confirm a contact.

Control, weaker proximity, tactical phase cues, camera reliability and selective
broadcast evidence form a temporal team-ownership path. Cross-camera matching
uses geometry rather than player IDs. A fixed H/A fallback supplies a team when
the subsequent activity model declares play live.

## 3. Physical and acoustic activity evidence

Eighteen features describe visibility, gaps, ball and player movement, preparation
and restart context, boundary position, contact, contest, phase, broadcast
visibility, pressure, tempo and geometric quality. The logistic fitting procedure
uses automatically generated physical LIVE/DEAD seeds from those features,
balanced class weights, L2=0.1 and 300 deterministic optimisation steps.
The fitted coefficients are held fixed while the restart features are updated.

Audio is extracted as mono PCM16 at 16 kHz. A 2048-sample Hann-window spectrum
with 320-sample hop searches for a pair of tones around 3765.625 Hz with a
460.9375 Hz separation. Local prominence, frequency stability and duration
produce automatic whistle proposals. The fixed minimum prominence is 16 dB;
minimum supported duration is 0.12 s. Long continuous tones are rejected.
A proposal is acoustic evidence, not proof that the referee stopped play.

A context-dependent acoustic signal supports stopping after live play or
restarting after established dead ball. Its fixed weight is 16. Preparation
and restart context contributes an additional weight of 2. Both halves use the
same settings and binary HMM switching costs of 15 in either direction.

## 4. Goalkeeper reception, holding and release

The goalkeeper module looks for an observed incoming ball, a projected arrival
just before a short occlusion, or compatible ball/keeper movement. Reception and
holding must connect to an observed outward release. Keeper location, nearby
field players, boundary evidence, new stopping cues and goal-kick preparation
constrain the episode. It is capped at 45 s and ends on loss of keeper evidence,
ball observation elsewhere, a stop cue or standard preparation.

An accepted chain lowers the DEAD evidence by up to 6 and assigns possession to
the goalkeeper's team when the activity decoder selects LIVE. Separate standard
preparation evidence raises DEAD evidence. This is a tracking-based hypothesis;
it is not visual recognition of a ball in the keeper's hands.

## 5. Confirmed restarts and tag retirement

Within each SET PIECES window the system looks for movement after a mostly
quiet or unobserved two-second period. The final configuration uses a 2.5 m/s
motion threshold and selects the first candidate confirmed by nearby contact,
at least 3 m of measured displacement along a connected trajectory, continued
motion and plausible field position. Confirmation can use up to 3 s of future
observations. Tracking gaps and implausible jumps cannot themselves confirm a pass.

After confirmation, preparation and set-piece activity context are disabled for
that event, and a three-second restart cue is placed at the detected boundary.
Overlapping tags remain independent. The acoustic signal and goalkeeper evidence
stay fixed in this configuration; the HMM still makes the final LIVE/DEAD decision.

## 6. Possession conventions and limitations

A missing ball is not automatically dead. Loose-ball team ownership follows the
temporal control path. Brief challenges can be smoothed away; observed proximity
is not necessarily established control. Keeper holding is treated as live control
when the reception/release chain and activity decision support it. Standard
preparation remains dead evidence until the restart mechanism closes it.

The system can confuse ball handling before a restart with actual play, miss a
new stoppage within a broad tag window, lose a keeper reception under occlusion,
or merge brief opponent possessions. Camera timing, replays, slot switches and
planar projection of airborne balls remain limitations. Scores are evidence
values rather than calibrated probabilities. This document describes runtime
behaviour; it makes no claim of independent accuracy or competition eligibility.

## 7. Reproduction

Use Python 3.11, the pinned requirements and ffmpeg/ffprobe. From the source root:

```bash
python -m possession --data-dir /path/to/match-data \
  --tactical-video /path/to/match-data/tactical/BAR-ATH_tactical.mp4 \
  --cache-dir /path/to/private-cache --output /path/to/output/msu114.csv
python evaluation/validate_submission.py /path/to/output/msu114.csv
```

Both halves include added time; the halftime interval is excluded. Export fills
the published pre-kickoff part of the second block with D and uses X outside the
published blocks. CSV rows are generated by code without subsequent editing.
Synthetic tests exercise the key behaviours; all data-dependent intermediate
files stay in the private cache. Starter loaders and evaluation tools originate
from the Metrica Sports repository under its retained MIT licence.
