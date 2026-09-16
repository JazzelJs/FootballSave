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


    

### 2026-09-16 — Stage 2 (position error vs SoccerNet ground truth)
- **What I did:** Decided that broken and missing PnLCalib cameras get filled in from the
  neighbouring frames. Wrote the empty cases of `match_frame` and chose the matching rule; Claude
  wrote the loop on my request and walked me through it line by line. Explained the worst frames
  from the minimap videos. *(Claude made `detect_track.py` / `to_pitch.py` / `minimap.py` run on
  SoccerNet clips, added the `jumpy()` camera rule, and changed the anchors in my `interpolate_h`.)*
- **What I learned (in my own words, from this session):**
  - Matching our dots to the true players: **closest pairs first, then skip the used ones.** If every
    player just takes its nearest dot in list order, an early player can steal the dot a later one
    needed, and then the score blames the detector for a mistake the matching made. That's "greedy":
    always take the best pair still available, never go back.
  - SNGS-043 frame 708, 2 players found out of 11: **they are celebrating the goal, not playing.** The
    detector only knows players in playing poses. It wasn't stacking: the biggest overlap between two
    people in that frame was 0.15, and the boxes are 40–87 px wide. After the goal the miss rate goes
    from 7% to 33%, but the median error doesn't move — a player we never find adds no error.
  - SNGS-043 frame 357: the camera pans right and there is a lot of jitter, but the main points are
    still good. **When all the dots shift together it's the camera, not the boxes.** Each dot moving
    on its own would mean the boxes. There the whole set moved 3.5 m together.
  - SNGS-028 frames 272–274: **it zoomed in from further away and the lines still match** — the dolly
    zoom (the Vertigo shot). Near the visible lines the picture hardly changes, so the fit still looks
    fine (3.3 px), but the grass far from those lines moves by meters, and that's where the players
    stand. Same lesson as my Stage 1 clicks: H is only pinned down where it has evidence.
  - **PnLCalib's own error is not a test**, now proven on one frame: frame 357 had the best
    self-reported fit in its neighbourhood (0.4 px) and the worst real error (3.7 m).
  - Interpolating H needs anchor points that are **in view in both frames**. The goal-area points were
    off-screen or behind the camera during a pan, which made the first fill-in 24 m off.
- **Numbers / results:**
  - (A) camera only, after filling in broken frames: SNGS-028 median 0.61 m (mean 1.10 → 1.09 m,
    was 3389 m before), SNGS-043 median 0.54 m, mean 0.62 m. Both clips now have a camera on all
    750 frames. `jumpy()` cut the worst case from 19.0 → 12.9 m and 15.3 → 7.9 m.
  - **(B) full pipeline: median 0.77 m (SNGS-028) and 0.56 m (SNGS-043)**, 95% 2.33 / 1.47 m.
    18% / 7% of true players missed (7% = up to frame 634, before the goal), 1.7 / 0.9 extra dots
    per frame.
  - **(B) − (A) = +0.16 m and +0.02 m.** Detection and tracking add almost nothing: the error is the
    camera. The labels' own wobble is 0.17–0.19 m, so that's the floor.
- **Still confused about / not answered yet:** why (B)'s biggest error is exactly 3.00 m while (A)'s
  is 12.9 m, and what that does to comparing the two. Also: if PnLCalib gave a perfect camera
  tomorrow, which file changes, `data/track/…` or `data/tracks/…`?
- **Next step:** smoothing each track and joining track pieces in meters, both measured against (B).
  The 2026-09-15 session (detection, tracking, ID switches, NMS) still owes its own entry.

### 2026-09-16 (part 2) — Stage 2 finished: smoothing, joining, teams
- **What I did:** Wrote `smooth_tracks` (Claude fixed how it wrote the result back). Decided the
  window by measurement, decided to cut SNGS-043 at the goal (frame 634) for the miss rate, and
  decided team colour should be per track, not per frame. Explained the worst frames from the minimap
  videos. Asked Claude to write `join_tracks`, `shirt_colour` and `assign_teams`, and to walk me
  through each line. Also had Claude read my other repo (Smart-Monitoring-System) to compare how that
  one tracks people.
- **What I learned (my own words where I said it, marked where it's Claude's wording):**
  - Smoothing = replace each position with the average of its neighbours in time. The wobble is
    random so it cancels; the player's real movement isn't random so it survives. Too long a window
    flattens real turns.
  - Frames 272–274: *"it zoomed in from further away, the lines still match"*. Claude's name for it:
    the dolly zoom (the Vertigo shot). Near the visible lines the picture barely changes, far from
    them the grass moves meters.
  - Joining track pieces: a track that ends and another that starts nearby in **meters** (not pixels,
    the camera pans) is the same player. Closest pairs first, skip the used ones — the same rule as
    matching dots to players.
  - *(Claude's wording, to rewrite in mine:)* (B) measures **where** a dot is, (C) measures **who** it
    is. Joining can only move (C), because renaming never moves a dot. And 69% of our ID failures are
    *swaps*, where the old id keeps living on another player, which joining can never fix.
  - *(Claude's wording:)* Teams from shirt colour: hue survives shadow where RGB doesn't, hue is a
    circle so use a histogram and let the bins wrap, and white/black kits have no hue at all, so
    saturation and brightness have to carry them.
  - My other project tracks people in two layers that never meet: BoT-SORT ids for line crossing
    (enter/exit), and face embeddings in Redis for returning customers. Nothing links a track id to a
    face, so it can't say *which* customer sat in which chair. Same lesson as (C) here.
- **Numbers / results:**
  - Smoothing, window 0.84 s: (B) median 0.77 → **0.70 m** (SNGS-028), 0.56 → **0.48 m** (SNGS-043).
    Picked by two numbers agreeing: (B) flattens out around 1 s, and our 95% player speed
    (5.5 m/s) matches the true players' 5.3 m/s — unsmoothed we "measured" 10.5 m/s.
  - Joining, 1 s / 3 m: our ids 210 → **132** and 161 → **122**; ids per real player 8.3 → **6.1** and
    6.4 → **5.5**; ID switches barely moved (375 → 359, 289 → 276) because most are swaps.
  - Teams: **96%** (SNGS-028) and **89%** (SNGS-043) of outfield player dots right. Referees called
    "other" 61% / 91%, goalkeepers 100% / 20%. On clip04 the keeper came out as "other" by colour alone.
  - Stage 2 final: position **0.70 / 0.48 m** median, 17% / 7% of players missed, **6.1 / 5.5** ids per
    real player, **96% / 89%** teams. (B) − (A) ≈ 0, so the error is the camera, not detection.
- **Still confused about / not answered yet:** why (B)'s biggest error is exactly 3.00 m when (A)'s is
  12.9 m, and what that does to comparing them. Whether `data/track/…` or `data/tracks/…` would change
  if the camera were perfect.
- **Next step:** Stage 3, the 3D viewer and the goalkeeper camera. The 2026-09-15 wrap-up is still owed.

### 2026-09-16 (part 3) — Stage 3: the 3D viewer, capsules, goalkeeper camera
- **What I did:** Asked Claude for the viewer skeleton, then asked for the capsule + team-colour
  change **line by line with an explanation for each line** before letting it write the code. Same
  for the goalkeeper camera. Spotted that clip04's id 14 is a **sideline referee**, not a bad
  position. Decided to skip the off-pitch filter after seeing the measurement, and to keep the
  viewer in the browser instead of porting it to SwiftUI/SceneKit for now. Reverted the
  interpolation once and had it re-applied.
- **What I learned (to rewrite in my own words — this is Claude's wording):**
  - A pool slot is not a player. The players list changes length and order between frames, so
    anything bound per slot (colour, and the same trap in interpolation) has to be re-bound per
    frame by **id**, not by position in the list.
  - A capsule is centred on its own origin, so a 1.8 m player stands at y = 0.9, not y = 0.
  - `MeshBasicMaterial` ignores lights. Flat colour was fine for a sphere; a capsule needs shading,
    which needs a light, or it renders black.
  - Two cameras cost nothing: `renderer.render(scene, camera)` takes the camera as an argument, so
    switching views is one variable. But `OrbitControls` is bound to one camera forever.
  - A `PlaneGeometry` is one-sided: from below the pitch is not dark, it is gone.
  - Interpolating between frames makes motion smooth without adding any information. The data is
    still 25 positions per second.
- **Numbers / results:**
  - Goalkeeper camera, 50° / 16:9 → half-FOV **39.7°**. A **fixed** aim from the goal keeps
    **100%** of players on screen in **all 750 frames** of SNGS-043 (worst frame too), so panning
    after the ball or the players would buy nothing. Median distance from the goal 42 m, where a
    1.8 m player is ~**28 px** tall on a 600 px view.
  - Players "in a gap" (id seen earlier, back later, missing now): SNGS-043 mean **2.0**, max 6,
    88% of frames; SNGS-028 mean **2.9**, max 9, 88%; clip04 mean **0.4**, max 2, 33%.
  - Off-pitch tracks (median |x| > 35 m): clip04 **1 of 23** (id 14, 308 frames, median |x| 35.9,
    always outside the touchline); SNGS-028 **0 of 132**; SNGS-043 **0 of 122**. Dropping them left
    every SoccerNet number identical, so I did not add the filter.
  - `teams.py` gave id 14 team **A in all 308 frames** — the linesman's kit is close to team A's.
    On SoccerNet referees were called "other" only 61% / 91% of the time, so this is the same
    weakness, but confident and wrong.
- **Still confused about:** _(mine to fill in)_
- **Decision gate, my answer:** all three of them matter, but **body pose first** → Stage 4.
- **Next step:** Stage 4, body pose. The 2026-09-15 wrap-up is **still** owed.
