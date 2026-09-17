# Football3D — Whiteboard Explanation

## The goal

Turn a normal football broadcast into a simple 3D football scene.

```text
football video  ───────────────►  3D pitch with moving players
```

The project answers three questions:

```text
Where is each player?  Who is each player?  What is each player doing?
        position              tracking              body pose
```

## The whole pipeline

```text
VIDEO
  ↓
CAMERA CALIBRATION
  ↓
PLAYER DETECTION + TRACKING
  ↓
PIXELS → PITCH METRES
  ↓
3D VIEWER
  ↓
BODY POSE + 3D BALL
```

## 1. Video becomes frames

A video is a sequence of images:

```text
video → frame 1, frame 2, frame 3, ...
```

We process each frame and then connect the results through time.

Our important test clips are:

- `clip04`: our football clip
- `SNGS-028`: SoccerNet ground truth
- `SNGS-043`: SoccerNet goal clip

## 2. Camera calibration

The video gives us pixels, but football uses metres.

```text
pixel (1200, 650)  →  pitch position (12 m, 35 m)
```

We use pitch lines and a homography to make this conversion for points on the
grass. A player's feet are approximately on the grass, so this works for
player positions.

It does not directly work for a head or a ball in the air, because those points
are above the grass plane.

Current position accuracy is roughly:

```text
SNGS-028: 0.70 m median error
SNGS-043: 0.48 m median error
```

The camera is currently the largest source of position error.

## 3. Detecting and tracking players

Detection finds a person in one frame:

```text
┌─────────┐
│ player  │
│         │
└────●────┘
     feet estimate
```

Tracking tries to keep the same ID over time:

```text
frame 1       frame 2       frame 3
 [166]         [166]         [166]
```

The difference is:

```text
detection = where is somebody?
tracking  = is this the same somebody?
```

Our detector and BoT-SORT tracker produce player boxes and IDs. We use the
bottom centre of each box as an approximate foot position.

## 4. Pixels become pitch positions

```text
box bottom centre → pixel coordinate → homography → pitch metres
```

The result is a top-down pitch map with moving dots.

We added:

- smoothing to reduce position wobble
- joining for short track fragments
- team colours from shirt crops
- gap display when a tracker temporarily loses a player

Team colour accuracy on SoccerNet is approximately 96% and 89%.

Identity is still imperfect: players can swap IDs when they pass close to one
another. Joining fragments helps, but it cannot fix a true ID swap.

## 5. The 3D viewer

The browser viewer currently shows:

- a 3D football pitch
- goals and pitch markings
- team-coloured player capsules
- smooth playback
- orbit camera
- goalkeeper camera
- missing-player gaps

The capsules show **where** players are, but not their:

- body posture
- facing direction
- arms and legs
- ball position in 3D

That is why body pose is the next stage.

## What we have learned

### Camera errors move everybody together

```text
good camera:       ● ● ● ●
bad camera:        →● →● →● →●
```

If all dots shift together, the camera is probably wrong. If one dot jumps by
itself, the box or track is probably wrong.

### Position and identity are different

```text
(B) position = where is the dot?
(C) identity = who owns the dot?
```

Renaming a track can improve identity scores, but it cannot move the dot or
improve its position error.

### Matching has a distance limit

Our full-pipeline matching ignores pairs farther than 3 metres. Therefore a
maximum error of exactly 3.00 metres is caused by the evaluation rule. Compare
the median and 95th percentile instead of comparing that maximum with the
camera-only maximum.

## Current stage: body pose

The current experiment focuses on SoccerNet clip `SNGS-043`:

```text
track ID: 166
frames:   560–620
box size: approximately 78 pixels tall
```

The planned flow is:

```text
player crop
    ↓
4DHumans / HMR2
    ↓
SMPL body pose
    ↓
24 3D joints per frame
    ↓
pose_166.npz
    ↓
3D viewer
```

This is inference, not training. We use a pretrained pose model and ask it to
estimate the player's pose.

The local SMPL model has been placed at:

```text
data/models/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl
```

The local SMPL scripts are:

- `src/pose/smpl.py`: loads and checks the SMPL model
- `src/pose/apply_smpl.py`: applies pose parameters and exports joints/vertices
- `notebooks/kaggle/stage4_pose_4dhumans.ipynb`: runs HMR2 on Kaggle

## SMPL and the Sketchfab character

They have different jobs:

```text
HMR2 / SMPL   = understands the pose
Sketchfab     = displays the character
```

The future flow is:

```text
estimated joints → character bone rotations → visible 3D footballer
```

The Sketchfab model is a display asset. SMPL or HMR2 provides the motion.

## The next checkpoint

We need to answer one simple question:

> Can a 78-pixel-tall broadcast player produce a believable 3D body pose?

We will check:

- whether the limbs are in sensible positions
- whether the pose stays stable between frames
- whether the player faces the running direction
- whether the pose flips or jitters
- whether the pose can be displayed in the viewer

If this works, we replace capsules with animated bodies. If it does not, we
keep capsules and use simpler signals such as running direction.

## Later: the ball

The ball is a separate problem:

```text
detect the ball → track it in pixels → estimate height/depth → fit a trajectory
```

A homography alone cannot locate a flying ball in 3D. We need camera parameters
and physical assumptions such as gravity, speed, and ball size.

## Core ML decision

Core ML is not needed for the current Kaggle experiment. It becomes useful if
we later want pose inference inside a native Mac or iPhone app.

The likely native pipeline would be:

```text
video crop → HMR2 Core ML model → SMPL joints → native 3D viewer
```

First prove that the pose is useful. Then measure whether local PyTorch/MPS is
too slow before adding Core ML conversion work.

## One-sentence summary

We are teaching a computer to turn broadcast pixels into pitch positions first,
then into recognizable 3D footballers, and eventually into a 3D replay of the
match.
