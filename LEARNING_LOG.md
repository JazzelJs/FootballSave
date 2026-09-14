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


    
