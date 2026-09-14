# Learning log

Written by me, in my own words. Claude only corrects mistakes.

## Template
### YYYY-MM-DD — Stage X
- **What I did:**
- **What I learned (explain like to a friend):**
- **Numbers / results:**
- **Still confused about:**
- **Next step:**


### 2026-09-14 — Stage 0 + Stage 1
- **What I did:** Set up uv, ffmpeg and the GitHub repo. Cut 7 clips from Barcelona vs
  Feyenoord. Wrote the pitch model, clicked 12 points on clip07 frame 0, computed H and
  checked it three ways (leave-one-out, overlay, top-down view).
- **What I learned (explain like to a friend):** H turns pixels into meters.
  *(Claude's correction: in our code H goes meters → pixels; `np.linalg.inv(H)` turns pixels
  into meters. Either way it is only right for points on the grass, z = 0.)*
- **Numbers / results:**
  - Stage 0: 7 clips, 1920×1080, 49.95 fps. Easy clip = clip07.
  - Stage 1 (clip07, frame 0): 12 points, leave-one-out error 4.10 px (previous session
    5.11 px). Top-down box width 40.3 m (real 40.32 m). Left edge of the frame drifts
    4–9 px (up to 0.5 m) because nothing could be clicked there.
- **Still confused about:** (skipped for now)
- **Next step:** Stage 1 moving-camera part: calibrate every ~10th frame of a panning clip,
  interpolate H in between.

Notes copied from Claude's explanations (not in my own words yet):

1. Players as streaks. "We change the point of view" is the right starting point, but "the points of the player stay there" isn't what happens. The precise version: H assumes every pixel shows a point on the grass.

A player's feet really are on the grass, so they land in the right spot.
His knees, chest and head are above the grass. For each of those pixels, H follows the camera's ray down to where it would hit the grass, which is behind the player.
The higher the body part, the farther back it lands: the head ends up meters behind the feet.

2. Why the points should be farther
    H is only pinned down where you clicked. Everywhere else it extends from those points, so small click errors grow with distance. box_front_left was 12 px off when H had no nearby points, and the frame's left edge drifts 4–9 px because you had no clicks there. With all points clustered, everything outside the cluster would be like that.


### 2026-09-14 — Stage 1 (moving camera)
- **What I did:** Clicked 14 frames of clip04 (every 25th). Wrote `interpolate_h` and `h_at` to
  guess H between keyframes, and tested the guess on the frames in between. Ran PnLCalib on all
  337 frames on Kaggle, wrote `pitch_to_soccernet` + `camera_to_h` to turn its cameras into our H,
  and compared it with my clicks. Exported `data/camera/clip04.json` for Stage 2.
  *(Claude fixed the last line of `interpolate_h` and the wrong goal end in `pitch_to_soccernet`.)*
- **What I learned (explain like to a friend):**
  - Interpolation is guessing between two things you know: km 0 at 12:00 and km 100 at 13:00 →
    km 75 at 12:45. It breaks if I stop or a traffic jam happens.
  - Averaging H depends on the scale, pixels don't. So we blend where the points land in the
    picture, not the numbers in the matrix.
  - PnLCalib beats interpolation in the first half because the camera changes speed there.
    PnLCalib sees each frame as its own. When the camera is steady they are the same.
  - *(Claude's corrections: players moving never changes H, only the camera's pan/tilt/zoom does.
    "The same" in the second half = a steady camera makes the straight-line guess almost right,
    so both land on the same ~5 px measuring floor: my click error + PnLCalib's detection error.)*
- **Numbers / results:**
  - My clicks (14 frames): own-fit 3.7 px; fair leave-one-out 4.9–9.6 px. `box_goalline_right` is
    the worst point in every frame (11–33 px) even though the clicks are right: it sits alone at
    the image edge.
  - Interpolation, keyframes 1 s apart: 15.1 px mean. First half 20–28 px (0.5–1.0 m on the
    pitch), second half 5–8 px. Keyframes 2 s apart: 21.8 px, and on frames 125/150 worse than
    not interpolating at all.
  - PnLCalib: 337/337 frames in 217 s on a Kaggle T4. 5.9 px mean vs my clicks (4–7 px on every
    frame, worst point 14.6 px). Its self-reported 4.4 px is not a test.
- **Still confused about:** (skipped for now)
- **Next step:** Stage 2: YOLO + ByteTrack on clip04, feet → meters with `inv(H)` from
  `data/camera/clip04.json`. Parked: PnLCalib jitter (check on the minimap) and the camera
  "moving" 8 m while zooming (fix before Stage 5).


    
