# PLAN.md — Football clip to 3D

Tags: `[YOU]` I write it · `[TOGETHER]` shared · `[CLAUDE]` boilerplate Claude writes.
Each stage: **Learn** → **Build** → **Checkpoint** (must pass) → **Explain it back**.

Rule of thumb: if a stage takes more than ~2x the estimate, stop and ask Claude
"what am I overcomplicating?" before continuing.

---

## Current status (updated 2026-09-17) — read this first in a new chat

**Where we are (facing direction now validated, 2026-09-18):** Stage 0 ✅ · Stage 1 ✅ · Stage 2 ✅ · **Stage 3: all the code is done ✅**
(viewer, capsules by team, orbit + goalkeeper cameras, gaps toggle, interpolation — see the stage
below) · **decision gate answered 2026-09-16: all three gaps matter, but body pose (Stage 4) goes
first** · **Stage 4 SMPL-only baseline complete: one male defender pose is in the viewer ✅; KTH
ground-truth validation is in progress.**
Where the pipeline stands, measured against SoccerNet ground truth: position **0.70 m / 0.48 m**
median (SNGS-028 / SNGS-043), 16–17% of players missed, **6.1 / 5.5** of our track ids per real
player, **96% / 89%** of outfield players given the right team. The remaining error is the camera,
not detection: (B) − (A) is about zero, and after smoothing even slightly negative.
Clips: **clip04** (our match, no ground truth) and the SoccerNet GSR clips **SNGS-028** (shot off
target) + **SNGS-043** (goal). Committed and pushed to GitHub (JazzelJs/FootballSave, `main`).
The four numbers the eval prints: **(A)** camera only · **(B)** full pipeline, where the dots are ·
**(C)** identity, whether a dot keeps its name · **(D)** teams.

**What happened on 2026-09-15/16 (not in `LEARNING_LOG.md` yet):**
- Downloaded SoccerNet GSR clips with `src/track/fetch_soccernet.py` (reads one clip out of the
  11.2 GB `valid.zip` on Hugging Face with byte-range requests, ~150–240 MB per clip). Picked
  028 + 043 by their event (`action_class` in each clip's labels). Both attack SoccerNet's right goal.
- Made `pitch_to_soccernet(xy, goal)` handle both ends: `GOAL_SIDE[clip]` in
  `src/calib/compare_pnl.py` (I wrote the right-goal branch: X = 52.5 − y, Y = −x). Checked on a
  picture (`outputs/axes_SNGS-033_00400.jpg`): +x = keeper's right = far touchline. ✅
- PnLCalib on Kaggle (`notebooks/kaggle/pnlcalib_soccernet.ipynb`, ~7.5 min per clip) →
  `data/camera/SNGS-0xx.json`. SNGS-028: 696/750 frames (fails 99–123 and 236–264; the labels show
  3.2 visible pitch lines there vs 8.8 on good frames — my "few lines" prediction was right).
  SNGS-043: 750/750. A few frames in both get a camera that is badly wrong (see below).
- **(A) camera-only error:** `uv run python src/track/eval_soccernet.py <clip>` = SoccerNet's own
  (perfect) boxes → foot pixel → OUR H → meters, vs the true meters. Detection/tracking play no part.
  | | SNGS-028 | SNGS-043 |
  |---|---|---|
  | **median** | **0.57 m** | **0.55 m** |
  | 95% | 3.18 m | 1.43 m |
  | labels' own wobble (median) | 0.17 m | 0.19 m |
  | mean | 3389 m ⚠️ | 1.53 m |
  The mean is useless: a few broken-camera frames (028: 265–276, 97; 043: 530, 397–398, 535)
  throw feet thousands of meters. Use the median / 95%, and deal with broken frames in (B).
  *(Claude wrote `camera_errors` (3 lines) on my request, short on time: walk me through it.)*
- **Broken / missing cameras are filled in from neighbours** (my choice, 2026-09-16). In
  `export_camera.py`, a PnLCalib frame is "broken" if it failed, its own error is > 10 px (normal ~4)
  or its focal length is < 1000 px (a view wider than ~90°, no broadcast camera does that). Those get
  H from `h_at` between the good frames around them, `"filled": true` in camera.json. Flags 0 frames
  on clip04, 64 on SNGS-028 (incl. the 54 failed ones), 11 on SNGS-043.
  First try was 24 m off on 028's long gap (99–123): `interpolate_h` blended the goal-area POINTS,
  which there sit 5,000–100,000 px off-screen or behind the camera. **Claude changed the anchors
  in my `interpolate_h`** to grass that is in view (pitch points under the lower half of both
  pictures) — the answer to its "does a point have to be visible in both frames?" question.
  clip04's Stage 1 numbers didn't move (15.1 / 21.8 px). Tested alternative: blending the camera
  itself (rotation, zoom, position) — 2.74 vs 3.73 m median on 99–123, equal on 1–2 s gaps; not
  used (more code), keep in mind if long gaps matter.
- **Second rule, `jumpy()` (added after I found frame 357):** a frame whose camera disagrees with the
  neighbouring frames by more than `MAX_JUMP` = 5 m on the grass is filled in too. `broken()` misses
  these: on SNGS-043 frame 357 PnLCalib's own error is 0.4 px (the *best* of its neighbourhood) with a
  camera 25 m too close and zoomed out — proof that its self-reported error is not a test. All dots
  shift together by 3.5 m there; per-frame the error went 3.70 → 0.42 m, frame 544 5.67 → 0.39 m.
  Flags 0 frames on clip04, 5 on SNGS-028, 7 on SNGS-043 (a bad frame drags its 2 neighbours in).
  | (A) camera only, after fill-in | SNGS-028 | SNGS-043 |
  |---|---|---|
  | frames with a camera | 750/750 (was 696) | 750/750 |
  | median / 95% / mean | 0.61 / 3.63 / **1.09 m** (mean was 3389) | 0.54 / 1.26 / **0.62 m** (was 1.53) |
  | max | 12.9 m (was 19.0 before `jumpy`) | 7.9 m (was 15.3) |
  | filled frames only, median | 2.64 m (long gaps, fast pan) | 0.57 m |

**The rest of 2026-09-16, in order (every step measured, all committed):**
1. **(B) full pipeline on SNGS-028 + SNGS-043:** ~~[CLAUDE] make `detect_track.py`, `to_pitch.py`,
   `minimap.py` work on SoccerNet clips~~ done 2026-09-16: frame names + fps per clip live in
   `src/track/clips.py`; the minimap now shows the whole pitch and, on SoccerNet clips, the true
   players as white rings. SNGS-028: 258 track IDs (165 shorter than 10 frames), 10.7 people per
   frame, the same as the truth; `MIN_CONF` dropped 48 tracks. SNGS-043: 202 IDs, ball in only
   18/750 frames. `match_frame` in `src/track/eval_soccernet.py`: greedy matching, closest pairs
   first, skip used ones, pairs > `MAX_DIST` = 3 m don't count (my idea; I wrote the empty cases,
   **Claude wrote the matching loop on my request, walked through line by line**). 4 self-checks
   incl. the A/B frame where matching player by player steals a dot.
   | | SNGS-028 | SNGS-043 |
   |---|---|---|
   | (A) camera only, median | 0.61 m | 0.54 m |
   | **(B) full pipeline, median / 95%** | **0.77 / 2.33 m** | **0.56 / 1.47 m** |
   | true players missed | 18% | 10% whole clip, **7% up to frame 634** |
   | extra dots per frame | 1.7 | 0.9 |
   | worst frames | 272–274, 106, 110 (all filled camera) | 540, 402, 533, 340, 146 |
   | most missed frames | 100–114 (filled camera) | 707–711 (celebration only, after `jumpy`) |
   **Frame 708 explained (me):** after the goal the players celebrate on the grass and the detector
   finds 2 of 11 — it only knows players who are *playing*. Not occlusion: measured max overlap 0.15,
   boxes 40–87 px wide, and only 2 raw boxes come out of YOLO (so not `MIN_CONF`, not the tracker).
   SNGS-043 misses jump 7% → 33% after the goal; the median error doesn't move (a player we never
   detect adds no error). My decision: report the football part too, `PLAY_END = {"SNGS-043": 634}`
   in `eval_soccernet.py` prints a second "(B) to 634" line. SNGS-028 has no such jump (19% → 17%).
   (B)'s max is exactly 3.00 m: pairs over `MAX_DIST` become missed + extra, so (B) vs (A) isn't a
   fair comparison of the tails — open question to me. **Next = [YOU] look at the worst /
   most-missed frames on the minimap and say WHY** (checkpoint), then commit.
2. ~~Decide how to handle broken-camera frames~~ filled in from neighbours (above).
3. **Smoothing done 2026-09-16** (I wrote `smooth_tracks` in `to_pitch.py`; Claude fixed how it wrote
   the result back: keep the player dict itself instead of searching the list with `frames.index`).
   Window is `SMOOTH_S` = 0.84 s, in seconds not frames (clip04 runs at 49.95 fps, SoccerNet at 25).
   | (B) median / 95% | SNGS-028 | SNGS-043 |
   |---|---|---|
   | no smoothing | 0.77 / 2.33 m | 0.56 / 1.47 m |
   | **0.84 s (21 frames)** | **0.70 / 2.14 m** | **0.48 / 1.14 m** |
   | 1.24 s | 0.72 / 2.21 m (worse) | 0.47 / 1.12 m |
   | 2.04 s | — | 0.49 / 1.28 m (worse) |
   Picked with two numbers that agree: (B) stops improving around 1 s, and our 95% player speed
   (5.5 m/s at 0.84 s) matches the true players' 5.3 m/s, where unsmoothed we "measured" 10.5 m/s.
   Misses also drop (1263 → 1140 on 043): a smoothed dot lands inside the 3 m matching limit more often.
4. **Identity number (C) added 2026-09-16** [CLAUDE], because (B) measures *where*, not *who*: a dot
   in the right place scores the same under any ID. Follows each true player (the labels carry a
   `track_id`) and watches which of our IDs sits on them: fragments per player, ID switches, and
   **merged ids** (one of our IDs on several people) — that last one is the guard against joining too
   greedily, not (B), which joining cannot move at all.
5. **Joining track pieces done 2026-09-16** (Claude wrote `join_tracks` in `to_pitch.py` on my
   request, walked through line by line; knobs `JOIN_GAP_S` = 1.0 s, `JOIN_DIST` = 3.0 m).
   | | SNGS-028 | SNGS-043 | clip04 |
   |---|---|---|---|
   | our IDs, before → after | 210 → 132 | 161 → 122 | 24 → 23 |
   | fragments per true player | 8.3 → 6.1 | 6.4 → 5.5 | — |
   | ID switches | 375 → 359 | 289 → 276 | — |
   | merged ids | 24 → 27 | 29 → 30 | — |
   | (B) median | 0.70 m (unchanged) | 0.48 m (unchanged) | — |
   **The honest finding: joining is not the cure.** 69% of ID switches on both clips are *swaps* —
   the old ID carries on somewhere else, i.e. the tracker trades IDs between two players who are
   both on screen. Joining can only fix the other 31% (the old ID really ended, median gap 1 frame).
   Bigger settings (2 s / 5 m) buy little more (fragments 5.2 on 043) at the same merge risk.
6. **Team colour done 2026-09-16** (Claude wrote `shirt_colour` + `assign_teams` in the new
   `src/track/teams.py` on my request, walked through line by line; I chose per-track over per-frame).
   Torso crop (25-55% of the box height, middle half of the width) -> drop grass pixels -> hue
   histogram (12 bins, saturation-weighted, smoothed around the circle) + median saturation and value
   -> median over 20 frames per track -> `cv2.kmeans` K=2, and a track further than 2x the median
   distance from its centre becomes "other". Written into `tracks.json` as `team`, drawn on the
   minimap (A yellow, B blue, other grey).
   | (D) teams | SNGS-028 | SNGS-043 | clip04 |
   |---|---|---|---|
   | outfield players given the right team | **96%** | **89%** | no ground truth; 12 / 9 / 2 tracks |
   | referees called "other" | 61% | 91% | — |
   | goalkeepers called "other" | 100% | 20% | the keeper is "other" ✅ |
   Two traps Claude hit and left comments about: a black referee kit is dark, so "drop dark pixels as
   shadow" deleted the referee; and `hue` is uint8, so `hue * 12` wraps at 255 and put blue in red's bin.

**Stage 3, what the viewer showed (2026-09-16, all committed and pushed):**
- The goalkeeper view works, and it exposes what the data does **not** have. At 24.8 s (the goal)
  nine capsules stand around the arc and you cannot tell what is happening: a capsule has no facing
  direction and there is no ball. At 28.0 s only three capsules are left — not a rendering problem,
  that is the detector losing the celebrating players (the measured 7% → 33% miss after frame 634).
- clip04's **id 14 is the sideline referee**: 308 of 337 frames, x −36.2 .. −35.4, never inside the
  touchline. Two things follow. (1) The left-edge homography is steady out there — the distance to
  the line does not drift over 6 s. (2) `teams.py` called them **team A in all 308 frames**, the
  same referee weakness as on SoccerNet (61% / 91% "other"), but confident.
- **Rejected, measured:** dropping tracks whose median |x| > 35 m. It fires on 1 of 23 tracks on
  clip04 and **0 of 132 / 0 of 122** on the SoccerNet clips, where every number stayed identical.
  A rule that never fires on the clips with ground truth cannot be validated — not added.

**Stage 4 baseline completed 2026-09-17:**
- **Track 166 is a defender, not the shooter.** The shooter is joined detector track **1131**
  (raw fragments 944 then 1131), confirmed against the video around frames 590–602.
- **The kick is frame 602.** Found from the ball's *pixel* speed (2.2 → 48.3 px/frame), not meters:
  the labels' ball `bbox_pitch` is the ground projection of a flying ball, so in meters the "speed"
  ramps smoothly 0 → 23 m/s over 35 frames and hides the kick completely. Keep for Stage 5.
- **The labels carry the ball in 738 of 750 frames, with pitch coordinates** — ground truth for the
  ball already exists on the SoccerNet clips.
- **Crop size, the number that decides Stage 4's ceiling:** id 166's box over frames 560–620 is a
  median **78 px tall, 40 px wide** (min 48, max 90). This stage's checkpoint assumed 100–200 px.
  Pose models want 256×256, so the defender gets upscaled ~3×. KTH (player fills a 480×640 frame)
  is the best case by a wider margin than planned.
- **Open, not answered:** do we need the real SMPL model files (free registration, academic
  licence), or is a joints-only 3D pose model enough for the checkpoint numbers (facing direction,
  reprojection, PA-MPJPE on KTH's 14 joints)? Separate question: a free rigged character (e.g. a
  Sketchfab CC model) is a *display* asset — it can be driven by joint rotations later, but it
  cannot produce pose and changes no number. Decide before downloading anything.
- **Also open:** GVHMR or 4DHumans. It changes the input: 4DHumans wants per-frame crops, GVHMR
  wants the video plus a track and also estimates camera motion. GPU work goes to **Kaggle**.
- **Kaggle result:** official 4DHumans/HMR2 processed **60/60 crops** for defender track 166,
  frames 560–619. The output has 24 joints and 6,890 vertices per frame in `data/pose/npz/pose_166.npz`.
  A small standard-library converter writes the 24 joints to `data/pose/json/pose_166.json` for the browser.
- **Viewer result:** the male SMPL mesh is anchored to track 166's pitch position and animated in
  the browser. The viewer reads `data/pose/mesh/pose_166.smpl`, generated from the newer NPZ with HMR2's
  rotation matrices and betas. The preview video is `data/pose/preview/pose_166_preview.mp4`. HMR2's exported
  Y axis needed flipping; the body is manually rotated 180° so this defender faces left. That
  facing correction is a display setting, not yet a learned orientation estimate. The viewer's
  Three.js imports now point to `Football3D/vendor/`.
- **Shooter result:** HMR2 processed track 1131 over frames 560–620 (frame 599 interpolated).
  The corrected SMPL export mirrors the wrongly assigned kicking leg, preserves the body turn visible
  in HMR2's preview, smooths yaw over 5 frames, and aligns frame 602 toward goal. Raw yaw changed by
  **6.4° median / 149.1° max per frame**; smoothing reduces that to **4.0° / 21.2°**.
- **Ground-anchor reprojection check:** track 1131's calibrated pitch point lands **11.6 px median**
  from the detector box's foot point over 42 visible frames (**5.5 px at frame 602**). This validates
  camera + pitch anchoring, not the still-open SMPL-joint reprojection checkpoint.
- **Ball:** calibrated pixel labels are shown in 3D while present. They stop at frame 590; no invented
  post-label flight is shown.
- **Facing-vs-running proxy:** over 32 moving samples, the adjusted pose differs from the track
  direction by **45.1° median**, **64.9° mean**, and **164.5° at the 95th percentile**. This is not
  a valid facing ground truth for this kick because the shooter is mostly planted; use image keypoints
  or KTH 3D ground truth before changing the orientation again.
- **KTH readiness:** `data/kth/sequence2/` is already present (175 frames, 3 cameras, 14 joints).
  Its supplied cameras reproject the supplied 3D joints to the supplied 2D labels at **5.7 px median**
  (**17.8 px at the 95th percentile**); use Camera 1 as the first HMR2 benchmark input.
- **KTH Kaggle run, completed 2026-09-17:** `sequence2.zip` was prepared at
  `data/kaggle/sequence2.zip` and attached as the Kaggle **kth dataset**. The notebook now finds the
  175 Camera 1 PNGs and the private SMPL model correctly. Its setup installs HMR2 from its own package
  metadata (rather than adding modules one by one), and its checkpoint cell uses HMR2's
  `download_models()` cache layout. HMR2 processed all **175/175** frames. The outputs are
  `data/models/kth_camera1_pose.npz` (24 joints, 6,890 vertices per frame) and
  `data/models/kth_camera1_preview.mp4` (512×256, 25 fps, 175 frames / 7.0 s). **KTH checkpoint
  complete:** `uv run python src/pose/eval_kth.py` maps the 24 HMR2 joints to KTH's 14 LSP joints.
  PA-MPJPE is **113.3 mm mean / 101.4 mm median** over 175 frames. Facing error is **17.4° mean /
  14.2° median / 42.9° at the 95th percentile**, after one sequence-wide similarity alignment. This
  is the close-up best case, not a claim for the much smaller broadcast players.
- **Broadcast SMPL reprojection check, completed 2026-09-17:** `yolo11n-pose.pt` detects 2D joints
  only after track 1131 is cropped and enlarged (at full frame it finds zero people: the shooter is
  ~70 px tall). `uv run python src/pose/reproject_smpl.py SNGS-043 1131` anchors the average SMPL
  ankle at the calibrated Stage 2 grass point and sends the 3D joints through PnLCalib's full camera.
  Across 39 usable frames / 457 detector-confident joint matches, the error is **19.7 px median /
  23.2 px mean / 46.0 px at the 95th percentile**. `outputs/smpl_reprojection_SNGS-043_1131_000602.jpg`
  visibly puts green SMPL joints close to red 2D detector joints at the kick. This is model-vs-model
  agreement, not labeled ground truth; occluded limbs and the 2D detector itself set its floor.
- **Multi-player pose, completed 2026-09-18:** `notebooks/kaggle/football3d_sngs043_approach_poses.ipynb`
  selects **00:00:15–00:00:25** of SNGS-043 (frames 375–625, ending at the goal). The resulting
  HMR2 pose files are `data/pose/npz/pose_284.npz` (250 frames), `pose_17.npz` (248), and
  `pose_171.npz` (221); their viewer meshes live in `data/pose/mesh/`. 171 is the joined player
  identifier: it correctly combines raw tracker fragments **663** and **1029**, rather than using
  raw track 171. The tracker has short gaps and ID swaps; the viewer uses a capsule in a gap rather
  than falsely joining a different player. The previous 244–369-frame attempt remains as
  `pose_17_approach_244_369.npz` for reference. (Absolute facing was not validated at that
  point; it is now — see the next entry.) A future Kaggle rerun needs the existing SoccerNet/raw-track
  dataset, private SMPL input, and private `hmr2_cache`; enable a T4 GPU.
**Absolute facing direction, done 2026-09-18:**
- **Two bugs, one of them mine to remember.** The viewer wrote `y' = footY - y`, negating a single
  axis. Flipping one axis is a **mirror**, not a rotation: it swaps left and right. That is what
  `--mirror-left-right` ("HMR2 assigned the kicking leg to the wrong side") and the manual 180° on
  track 166 were really compensating for — HMR2 was right, the viewer was mirroring it.
- **Facing is not estimated, it is a change of basis.** HMR2 hands back a body in the coordinates of
  the camera that saw it, and PnLCalib already fits the rotation from that camera to the pitch. New
  `src/pose/orient.py`: `WORLD_TO_SCENE[goal] @ R_pnl.T @ crop_to_full(K, box centre)`, all three
  determinant +1, so nothing mirrors. `crop_to_full` is the CLIFF correction: HMR2 sees a *crop*, so
  its body sits in a camera aimed at the crop centre, not down the optical axis.
- **Measured against the direction of travel** (`orientation_error.py`, rewritten). A running player
  faces where they run, so motion is a usable stand-in — but only while running, which is why the old
  45° number on the planted shooter measured nothing. `--min-speed 3.0` m/s:
  | facing error, median / 95% | track 284 | track 17 | track 171 |
  |---|---|---|---|
  | samples above 3 m/s | 165 of 250 | 160 of 248 | 68 of 221 |
  | **pitch space (new)** | **10.4° / 66.6°** | **15.2° / 52.0°** | **29.9° / 104.1°** |
  | camera space, best possible zero | 16.8° / 68.2° | 21.2° / 107.3° | 53.2° / 177.2° |
  The camera-space row is given the offset that *minimises its own error*, so it cannot blame tuning.
  284 and 17 land near KTH's close-up 17.4°. Track 171 is worse and its five biggest errors are
  frames 568–572, right where the joined track hands over — the join, not the rotation.
- **Two things I predicted and got wrong, both checked:** (1) smoothing yaw in pitch space would beat
  smoothing it in camera space, because a moving average in camera space also averages in the pan.
  Measured per-frame yaw change: camera 3.4° vs pitch 3.6° on track 284 — the same. The pan is only
  ~0.2°/frame, so it corrupts the *absolute zero*, not the frame-to-frame jitter. Windows 1 to 15
  move the median by under 2°; 5 is kept and is nearly a no-op. (2) The shooter's 1.2° at the kick
  looked like a win, but he has only **4 samples above 3 m/s** in 60 frames — he is planted, so his
  number is noise either way. The three running players are the evidence.
- **Visual:** `src/pose/draw_facing.py SNGS-043 602 284 17 171 1131` draws a 2 m yellow facing arrow
  on the grass plus a red direction-of-travel arrow. `outputs/facing_SNGS-043_000602.jpg`: the
  shooter faces the goal at the kick, the two arrows agree on the running players.
- **Deleted:** `--stabilize-yaw`, `--mirror-left-right`, `--align-yaw-frame`, `rotationY`, the
  `poseRotation` query override. Re-export with `--clip SNGS-043 --track <id> --smooth-yaw 5`.
- **Open:** the travel-direction proxy is itself noisy (tracker jitter over 4 frames), so 10° is a
  ceiling on what it can prove, not the true facing error. KTH is the only real ground truth.
- **Decision:** use SMPL directly for now. Do not spend the next step retargeting the Quaternius or
  Sketchfab character; a display character would add work without improving the pose estimate.

**Start the next session with:**
1. **Next Stage 4 task:** track 171's facing is 30° median against 68 running samples, and its worst
   frames (568–572) are the hand-over inside the joined track. Check whether the join is putting two
   different players under one id there — that is the Stage 2 team-colour idea (item 4) coming back.
2. **Wrap-up still owed for 2026-09-15** (CLAUDE.md rule 5): detection vs tracking, ID switches, NMS,
   *where* vs *who*, foot point → meters, wobble, goal side. 2026-09-16 has its `LEARNING_LOG.md`
   entry; parts of it are Claude's wording and I should rewrite those in my own words.
3. Open questions I haven't answered yet (no rush, they're small):
   - (B)'s biggest error is exactly 3.00 m while (A)'s is 12.9 m. Why, and what does that do to
     comparing (B) with (A)?
   - If PnLCalib gave a perfect camera tomorrow, which file changes: `data/track/…` or `data/tracks/…`?
4. Left in Stage 2 on purpose, to pick up when it matters: use the team colours to refuse a join or a
   tracker hand-over between two different kits (69% of ID switches are swaps, which joining can't fix).

**Pipeline for a clip, as it runs today** (all from the repo root, all local on the Mac):
1. Frames: `src/extract_frames.sh "<source video>" clip04 00:02:09 00:02:15.74` →
   `data/frames/clip04/00000.jpg …` (source video + times in `data/clips/README.md`).
2. Camera per frame: PnLCalib on Kaggle (`notebooks/kaggle/pnlcalib_clip04.ipynb`) →
   `data/pnlcalib/pnlcalib_raw_clip04.json` → `uv run python src/calib/export_camera.py clip04` →
   `data/camera/clip04.json`. Checked against my clicks with `src/calib/compare_pnl.py clip04`.
3. Detection + tracking: `uv run python src/track/detect_track.py clip04` (~1 min on MPS) →
   `data/track/clip04_football-player-detection-v9_botsort.json` (raw boxes, pixels).
4. Feet → meters: `uv run python src/track/to_pitch.py clip04` → `data/tracks/clip04.json`
   (= `tracks.json`). Also smooths each track over `SMOOTH_S` = 0.84 s and joins track pieces
   (`JOIN_GAP_S` = 1 s, `JOIN_DIST` = 3 m). Prints check 1 (clicks → meters) and check 2 (top speed
   per ID). `to_pitch.py clip04 0` turns smoothing off, any other number = window in frames.
5. Teams: `uv run python src/track/teams.py clip04` → fills `team` in the same `tracks.json`
   ("A" / "B" / "other"). Run it after `to_pitch.py`, which resets `team` to null.
6. Minimap: `uv run python src/track/minimap.py clip04` → `outputs/minimap_clip04.mp4` (dot = team,
   trail = track id, white rings = ground truth on SoccerNet clips).
   Boxes-only video for comparing detectors/trackers: `src/track/draw_tracks.py <raw json>`.
7. Watch it in 3D: `python3 -m http.server 8000` **from the repo root**, then
   `http://localhost:8000/src/viewer/?clip=clip04`. Buttons: goal view (`c`), show gaps (`g`).
   After editing the page, hard-reload (Cmd-Shift-R) or the browser serves the old one.

**SoccerNet GSR clip (ground truth), as it runs today:**
1. `uv run python src/track/fetch_soccernet.py` (list clips) → `… fetch_soccernet.py SNGS-028` →
   `data/soccernet/SNGS-028/img1/000001.jpg …` + `Labels-GameState.json`. Add the clip to
   `GOAL_SIDE` in `src/calib/compare_pnl.py`.
2. Zip `<clip>/img1` → Kaggle dataset → `notebooks/kaggle/pnlcalib_soccernet.ipynb` →
   `data/pnlcalib/pnlcalib_raw_<clip>.json` → `uv run python src/calib/export_camera.py SNGS-028`.
3. Steps 3–5 of the clip pipeline above work on SoccerNet clips too (`detect_track.py SNGS-028`
   ~3 min, `to_pitch.py SNGS-028`, `minimap.py SNGS-028` with the true players as white rings).
4. `uv run python src/track/eval_soccernet.py SNGS-028` → the four numbers: **(A)** camera only
   (their perfect boxes through our H), **(B)** full pipeline (where our dots are, plus missed and
   extra), **(C)** identity (our ids per real player, ID switches, merged ids), **(D)** teams.

**Known issues, parked on purpose (and where each one gets fixed):**
- **Track IDs are still not one-per-player:** 24 IDs for ~20 people on clip04. Causes: extra
  boxes where two players overlap (#113, #171, #19, #265 — #265 "runs" 12.9 m/s), and the
  goalkeeper lost at frame 228 → back as #250 at 262. Longer tracker memory and appearance re-ID
  did NOT help (tested). Planned fix = "join track pieces in meters" (Stage 2 → Build), measured
  against SoccerNet.
- **Wobble:** 0.17 m per dot on average (95% under 0.46 m, max 1.49 m). Split: camera part
  (all dots move together, PnLCalib) 0.10 m, box part (each dot alone) 0.14 m, all big jumps are
  boxes. Fixes: smoothing per track (Stage 2, measured vs SoccerNet), ankles instead of box
  bottom (Stage 4), steadier camera (before Stage 5).
- **PnLCalib camera "moves" ~8 m while zooming** (zoom vs distance trade-off). Fix with one fixed
  camera position per clip, only pan/tilt/zoom per frame — before Stage 5.
- **Class labels of the football detector are unreliable on our match:** the referee is called
  "player" 97% of the time, a Barcelona player "referee" 90 times, the goalkeeper right only
  about half the time. Use it for *where*, not *who*. `team` in `tracks.json` is `null` →
  team classification is still needed before Stage 3 (team colours).
- **Ball:** 2–4 "ball" boxes in 255 of 337 frames (spare balls by the ad boards, false boxes),
  so "ball found in 296 frames" is too optimistic. `ball_px` = most confident box. Stage 5.
- **`MIN_CONF = 0.5`** in `src/track/to_pitch.py` (drop person tracks with a lower mean YOLO
  confidence) was picked on clip04 alone: junk 0.20–0.43, real players ≥ 0.60. Re-check the gap
  on other clips/recordings.
- **Touchlines** on the minimap assume a 68 m wide pitch (±34 m): drawing only, never measured.
- ~~clip04-only `pitch_to_soccernet`~~ Fixed 2026-09-15: `GOAL_SIDE[clip]` in `compare_pnl.py`
  ("left"/"right" = SoccerNet X = ∓52.5), checked on SNGS-033 frame 400 with the label positions
  (`outputs/axes_SNGS-033_00400.jpg`). A new clip needs its line in `GOAL_SIDE`.
- **Questions I skipped on 2026-09-15/16 (optional):** where on a pitch does the camera see the
  fewest lines? Will (A) be bigger or smaller than clip04's 0.35 m, and why? Why can't
  `rep_err_px` alone be trusted? Why does the right goal put the keeper's right at −Y?
- **Questions I skipped (optional to revisit):** is the worst check-1 point (0.98 m) far from
  the camera or near, and why? What is #265 at 12.9 m/s around frame 325? How far behind a player
  does a knee-height box put him (head at 1.8 m → 6.6 m behind)?

---

## Stage 0 — Setup and choosing clips  (~1 session)

**Learn**
- Why clip choice matters: camera cuts, replays and close-ups break calibration.
- What frame rate means for a fast ball (at 30 m/s and 25 fps ≈ 1.2 m per frame).

**Build**
- [CLAUDE] ~~Project setup with `uv`, `.gitignore`, folder structure.~~ Done: Python 3.12,
  `pyproject.toml` + `uv.lock`, git repo on GitHub.
- [CLAUDE] ~~`ffmpeg` script to cut a clip and extract frames.~~ Done:
  `src/extract_frames.sh <video> <clip> <start> <end>` → `data/frames/<clip>/00000.jpg`
  (file number = `frame` in `tracks.json`), prints fps and resolution.
- [YOU] Pick **3–5 clips**, each: one continuous camera shot (no cuts), 5–10 s, the
  penalty box clearly visible, a shot on goal at the end. Write in `data/clips/README.md`
  for each clip: start/end time, fps, resolution, how much the camera moves, **camera
  angle** (e.g. "main cam, halfway line" / "behind goal, high"), and which goal is attacked
  (left or right of screen).
  Different angles *between* clips are fine (each clip gets its own H); an angle change
  *within* a clip is not.
- ~~[YOU] Download one SoccerNet GSR validation clip.~~ Deferred to Stage 2 (first used there).

**Checkpoint**
- Frames extracted; you can state the fps and resolution of each clip.
- You have one clip where the camera barely moves ("easy clip") — start with that one.

**Explain it back:** Why is a single continuous shot important? What breaks if the
clip contains a replay?

---

## Stage 1 — Camera calibration by hand  (~2–3 sessions)
The foundation for everything else. Do it manually before using any library.

**Learn**
- Pinhole camera: a pixel is a *ray*, not a point.
- Homogeneous coordinates and why a 3x3 matrix can map a plane to an image.
- Homography: why it needs ≥ 4 point pairs, and why it's only valid for points ON the
  plane (the grass) — not for heads or a ball in the air.
- Pitch dimensions: use the penalty box (16.5 m deep x 40.32 m wide), goal area
  (5.5 m x 18.32 m), penalty spot (11 m). These are standardized. Full pitch length
  and width vary between stadiums, so don't rely on them.

**Build**
- [CLAUDE] A small click tool (OpenCV or matplotlib) to click points on a frame and save
  them to JSON.
- [YOU] `src/calib/pitch_model.py`: 3D coordinates (x, y, z=0) of penalty box corners,
  goal area corners, penalty spot, goal posts, in the project pitch frame.
  Name points by the **goalkeeper's** left/right (`+x` = GK's right), not the screen's —
  when the attacked goal is on the other side of the screen, left/right on screen flip.
  A mislabeled corner gives a mirrored pitch that can still look plausible.
- [YOU] `src/calib/homography.py`: compute H with `cv2.findHomography` from your
  clicked points.
- [YOU] Convert a clicked foot position (pixel) → pitch meters.
- [TOGETHER] Draw the pitch model lines back onto the frame using H (reprojection overlay).
- [TOGETHER] Warp the frame to a top-down view with `cv2.warpPerspective`.
- [YOU, stretch] Implement the homography yourself with DLT + SVD in NumPy and compare
  with OpenCV's result.
- [YOU, stretch] **Full camera (K, R, t) against ground truth**, using `SNv3D.csv` from
  SoccerNet-v3D (see Resources — no images needed). Build K, R, t from a row's
  `calibration`, project `ball_3D` into the image, compare with the `ball_bbox` center.
  Your error should match their `rep_error` column. Stage 5 needs this full camera model.
  Watch out: SoccerNet's world frame has its origin at the **center spot** (and, I think,
  `z` pointing down — check the sign of `position_meters[2]`). Write the conversion to
  our pitch frame and test it on one point you can reason about (e.g. the center spot).

**Checkpoint**
- Reprojected pitch lines sit on the real lines. Mean reprojection error **< 5 px** on
  points you did NOT use to compute H (hold some out!).
- Top-down view: penalty box looks rectangular with correct proportions.
- Per clip: note the camera angle next to the error. Is error worse for some angles?

**Explain it back:** Why can't this homography tell you where the ball is when it's in
the air? What happens to accuracy if all your clicked points are close together? Why
is 1 px of click error worth more meters on the far side of the box?

**Moving camera (after the easy clip works):**
- ~~[YOU] Calibrate every ~10th frame by hand, interpolate H in between. Measure how bad
  interpolation is on a frame you calibrated but held out.~~ Done on clip04: every 25th frame
  (50 fps), `src/calib/interpolate.py`. 15.1 px with keyframes 1 s apart, 21.8 px at 2 s.
- ~~[TOGETHER] Then try an automatic method (PnLCalib or No-Bells-Just-Whistles) in Colab,
  and compare it with your manual calibration on the same frames.~~ Done: PnLCalib on Kaggle
  (`notebooks/kaggle/pnlcalib_clip04.ipynb`), 5.9 px vs clicks (`src/calib/compare_pnl.py`).
  Used for `data/camera/clip04.json` (`src/calib/export_camera.py`). New clips: run PnLCalib,
  click 3–4 frames to check it. `pitch_to_soccernet` only handles "goal on the left" so far.

---

## Stage 2 — Detection, tracking, positions on the pitch  (~2–3 sessions)

**Learn**
- Detection (per frame boxes) vs tracking (same ID across frames). What an ID switch is.
- Why the bottom-center of a bounding box is only an approximation of the feet.

**Build**
- ~~[CLAUDE] Run Ultralytics YOLO + ByteTrack locally with `device="mps"`, save raw boxes.~~ Done
  on clip04: `src/track/detect_track.py` (needs `ultralytics` + `lap`, both in `pyproject.toml`).
  What we tried, all on clip04 (337 frames), numbers = different person IDs (~20 real people):
  | Detector | Tracker | Person IDs | Ball found | Notes |
  |---|---|---|---|---|
  | `yolo26m.pt` (COCO "person") | ByteTrack | 48 | 61 | finds photographers/staff by the pitch |
  | Roboflow `football-player-detection-v9.pt` | ByteTrack | 85 → **32** | 274 | 85 until `agnostic_nms=True`: it put a "player" AND a "goalkeeper" box on the same person in 171/337 frames |
  | same | ByteTrack, `track_buffer` 30 → 150 | 32 | 274 | output identical: lost players don't come back where the tracker predicts |
  | same | **BoT-SORT** (camera-motion compensation) | 32 | 296 | **chosen** (default in the script) |
  | same | BoT-SORT + re-ID (appearance) | 33 | 301 | teammates wear the same kit |
  The ~12 IDs that start mid-picture appear at the same frames with every tracker → they come from
  the detector's boxes, not the tracker. Roboflow model: YOLOv8x, fine-tuned 100 epochs at 1280 px
  on Bundesliga broadcast frames; classes `ball, goalkeeper, player, referee`; ~60 s for 337 frames.
- ~~[YOU] Foot point = bottom-center of each box → pitch meters using Stage 1 H.~~ Done:
  `foot_point` + `pixels_to_meters` (= `project(inv(H), pixels)`) in `src/track/to_pitch.py`.
  Check 1 (my 169 clicked pitch points on 14 frames → meters): **0.35 m mean, 0.98 m worst.**
  Check 2 (top speed per ID, every 10th frame): real tracks 2.9–10.4 m/s.
- ~~[YOU] Write `tracks.json` (format below).~~ Done: `frame_entry` in `src/track/to_pitch.py` →
  `data/tracks/clip04.json`. Tracks with mean confidence < 0.5 are dropped (5 IDs = a pile of
  towels by the goal post, 1 = a steward 7 m behind the goal, 2 tiny blips) → 24 person IDs.
- ~~[CLAUDE] 2D minimap video (top-down pitch with dots) next to the original frame.~~ Done:
  `src/track/minimap.py` → `outputs/minimap_clip04.mp4` (colour per ID, 1 s tail per dot).
  I watched it: positions look right, the jitter is clearly visible (measured: see Current status).
- ~~[YOU] Download one SoccerNet GSR validation clip.~~ Done 2026-09-15 with
  `src/track/fetch_soccernet.py` (Hugging Face `SoccerNet/SN-GSR-2025`, `valid.zip` = 11.2 GB, so
  the script reads only one clip's bytes out of it, ~150–240 MB per clip). 58 clips, each 750
  frames (30 s, 25 fps) + `Labels-GameState.json`; the event per clip is in the labels' `info`.
  Chosen: **SNGS-028** (shot off target at frame ~388) and **SNGS-043** (goal at ~619). SNGS-033
  (foul, first try) is downloaded too but not used. Labels: per person `bbox_image` (px) +
  `bbox_pitch` (SoccerNet meters, origin centre spot, +Y = near side), pitch lines per frame,
  **no camera parameters** → PnLCalib on Kaggle. Whole 30 s, not a steady stretch: PnLCalib gives
  a camera per frame, so report the error per part of the clip instead.
- [YOU] Evaluate on the SoccerNet GSR clip: match your players to ground truth and
  compute mean position error in meters. **(A) camera only done** (`src/track/eval_soccernet.py`,
  median 0.57 / 0.55 m on SNGS-028 / 043, see Current status); **(B) full pipeline next.** Look at the worst cases and say WHY they're bad.
  To run our pipeline on the GSR clip it needs: frames → camera per frame (check whether the GSR
  labels include camera parameters or only pitch lines; otherwise PnLCalib on Kaggle) → `detect_track.py` → `to_pitch.py`. `to_pitch.py`,
  `minimap.py` and `export_camera.py` currently hard-code clip04 paths / its goal side.
- [YOU] Once the error number exists, try the two fixes and keep what the number says helps:
  (a) **smoothing** each track's positions over ~0.2–0.3 s (try a few window sizes; too long
  rounds off real sharp turns), (b) **joining track pieces in meters**: a track that ends and
  another that starts close by (in meters, not pixels — the camera pans) a few frames later =
  the same player. Not done yet on purpose: without ground truth we'd only be tuning by eye.
- ~~[YOU, needed before Stage 3] Team for each track.~~ Done 2026-09-16, `src/track/teams.py`:
  shirt colour of the torso crop, clustered into 2 teams + "other". 96% / 89% of outfield player
  dots right on SNGS-028 / SNGS-043 (see Current status).
- [TOGETHER, optional] Run the full sn-gamestate baseline in Colab on the same GSR clip
  and compare with your simpler pipeline. Where does the baseline win, and why?

**Checkpoint**
- ~~Minimap video looks right.~~ Done on clip04 (2026-09-15), jitter noted and measured.
- ~~A measured number: mean position error on the GSR clip, e.g. "1.4 m, worst 4 m at
  frame 212 because of occlusion".~~ Done 2026-09-16: **median 0.77 m on SNGS-028, 0.56 m on
  SNGS-043** (95%: 2.33 / 1.47 m), 18% / 7% of true players missed. Why the worst frames are bad:
  - **SNGS-043 708–711** (9 of 11 missed): the goal has been scored and the players are celebrating
    on the grass. The detector only knows players who are *playing*; the 2 it finds are the only 2
    still standing. Not occlusion: max overlap 0.15, boxes 40–87 px wide, only 2 raw boxes from YOLO.
  - **SNGS-043 357 and 544** (3.7 / 5.7 m, all dots shifted together): PnLCalib put the camera 25 m
    off for a single frame, with its *best* self-reported error (0.4 px). Fixed by `jumpy()` → 0.4 m.
  - **SNGS-028 99–123** (up to 10 of 14 missed): the camera pans across midfield, so only 2–4 pitch
    lines stay in view, PnLCalib finds no camera at all, and the filled-in one is a 1 s guess
    across a moving camera → ~3–4 m off, and most dots land further than 3 m from their player.
  - **SNGS-028 270–280** (4.2–4.6 m): dolly zoom. PnLCalib moves the camera 40 m back and zooms in
    (focal 4500 → 7861); near the 5 visible lines the picture hardly changes (3.3 px fit), but the
    grass far from them shifts by meters. A whole stretch drifts together, so `jumpy()` can't see it
    and only the ground truth can. Fix = one fixed camera position per clip, parked before Stage 5.

**Explain it back:** Where does position error come from? (Calibration? Box bottom?
Occlusion?) Which one is biggest in your data?

---

## Stage 3 — 3D capsule viewer + goalkeeper camera  (~2 sessions)

**Learn**
- Coordinate system conversion (pitch z-up → three.js y-up).
- Interpolating positions between frames; camera field of view.

**Build** — all done 2026-09-16, one file: `src/viewer/index.html` (three.js 0.170 from a CDN
importmap, no build step, no new dependency). Serve the **repo root** (`python3 -m http.server 8000`)
and open `/src/viewer/?clip=SNGS-043` — `file://` fails, fetch is blocked there.
- ~~[CLAUDE] three.js viewer skeleton~~ pitch from `pitch_model.py`'s constants (both ends + goals),
  loads `tracks.json`, play/pause/scrub. Pitch (x, y) → three.js (x, up, −y), so the attacked goal
  is at z = 0. Playback steps by real time, so 25 fps and 49.95 fps clips both run at the right speed.
- ~~[YOU] Player capsules (1.8 m tall), team colors, update each frame.~~ Coloured from `team` in
  `tracks.json`, same colours as `minimap.py`. The material is re-assigned every frame: a pool slot
  is not a player.
- ~~[YOU] Two cameras~~ orbit (stops at grass level) + goalkeeper at (0, 1.7, 0), looking level up
  the pitch. **Fixed aim on purpose**: a 50° / 16:9 view from the goal keeps 100% of players on
  screen in all 750 frames of SNGS-043, so panning would buy nothing. No ball in meters to look at
  anyway (`ball_px` only, 18/750 frames).
- ~~[TOGETHER] Toggle to show which players were never visible~~ "show gaps" (or `g`): a faint
  capsule where a lost player was last seen. In a gap = the id was seen earlier and comes back
  later, but is missing now. Mean 2.0 per frame on SNGS-043 (max 6, 88% of frames), 2.9 on
  SNGS-028, 0.4 on clip04. It stands still on purpose — a line to where they reappear is a guess.
- Extra: positions interpolate between frames (matched by id), with a self-check that runs on load.

**Checkpoint**
- ~~You can watch the clip in 3D from the goalkeeper position.~~ Done 2026-09-16 (`1751ac5`,
  `313262e`, `a1c3f37`).

**Decision gate (answered 2026-09-16):** The goalkeeper view is useful for camera framing, but
capsules do not show facing or body action. All three gaps matter; body pose comes before the ball.

---

## Stage 4 — Body pose from a single camera  (~3–4 sessions)

**Learn**
- SMPL body model: pose (joint rotations) + shape (body proportions).
- Rotation representations: axis-angle, rotation matrices, quaternions.
- Root-relative pose vs global position — and why we take position from Stage 2
  (the pitch plane), not from the pose model.
- Known weaknesses: depth flips, invented occluded limbs, jitter, wrong facing direction.
- **2D reprojection error is blind to depth flips:** a leg flipped toward/away from the
  camera lands on the same pixels. You need 3D ground truth (or a second view) to see it.
- 3D pose metrics: MPJPE (mean joint distance) vs PA-MPJPE (after aligning rotation,
  scale and translation — measures pose shape only).

**Build**
- ~~[TOGETHER] Run HMR2 on one tracked player and export SMPL joints.~~ Done 2026-09-17 on Kaggle
  for defender track 166, frames 560–619.
- ~~[TOGETHER] Show the joints in the viewer.~~ Done 2026-09-17 with a browser-readable JSON
  export, pitch anchoring, close pose view, and orbit-view restore.
- [TOGETHER] Import the Sketchfab character, inspect its armature, and retarget the SMPL joints.
- [TOGETHER] **Kaggle** notebook (not Colab): run a single-view model (e.g. GVHMR or 4DHumans) on
  crops of 1–3 key players (the shooter first — **not identified yet**; track 166 turned out to be
  a defender, see Current status). Needs SMPL model files (free registration) — or a
  joints-only model instead, still to be decided.
  Export per-frame pose to `pose_<track_id>.npz`.
- [YOU] Anchor each body: root position = Stage 2 pitch position, pose from the model.
- [TOGETHER] Show skeletons/meshes in the viewer.
- [YOU] **Reprojection check:** project 3D joints back onto the frame using the camera,
  measure joint error in pixels vs 2D keypoints.
- [YOU] **Orientation check:** compare the body's facing direction with the running
  direction from Stage 2 tracks. Report the angle difference over time.
- [TOGETHER] **3D error on real footballers (KTH Football II, see Resources).** Run the same
  pose model on one camera of one KTH sequence (start with the sliding tackle), map SMPL
  joints → KTH's 14 joints, compute PA-MPJPE vs the 3D ground truth. Claude: Colab run +
  `.mat` loading. You: joint mapping + the metric.
- [YOU] **Facing direction vs ground truth (KTH):** facing = normal of the hip line in the
  ground plane. Compare model vs GT per frame, report mean angle error. This checks the
  running-direction proxy above: how good a guess was it?
- [CLAUDE, stretch] Smoothing filter; retarget to a rigged character in Blender.

**Checkpoint**
- Shooter's pose reprojects within a reasonable pixel error (report the number).
- PA-MPJPE on KTH (report the number). Treat it as a **best case**: KTH players fill the
  frame, your broadcast players are far smaller (track 166: median 78 px tall, not the 100–200 px
  this plan assumed).
- You can state how wrong the facing direction is on average, vs KTH ground truth and
  vs running direction on your clip.

**Explain it back:** Why is position taken from the pitch and not the pose model?
Why is facing direction the error that matters most for a goalkeeper? Why can a pose
with 4 px reprojection error still be badly wrong in 3D?

---

## Stage 5 — The ball in 3D  (~3 sessions)

**Learn**
- Why one camera can't locate a flying ball (the ray problem).
- Priors that help: kick point on the ground, gravity + drag, known ball size (~22 cm).
- Depth from apparent size: `distance ≈ f · 0.22 / d_px`. Fragile — a far ball is ~10 px
  wide, so 1 px of box error ≈ 10% ≈ several meters. SoccerNet-v3D reports 4.2 m mean error
  for this per-frame method, and 6–14 m error from a 10% size error.
- Least-squares fitting; why the answer is only as good as the priors.

**Build**
- [CLAUDE] Ball detection with SoccerNet-v3D's `yolo-sn-ball-opt.pt` (YOLOv11, already
  fine-tuned on broadcast football — see Resources). Measure the **detection rate** on your
  clips (% of frames with the ball found). The paper shows these models transfer poorly to
  unfamiliar camera setups, so measure, don't assume. Fall back to YOLO "sports ball" or
  fine-tuning only if the rate is bad.
- [YOU] **Size-based baseline on ground truth:** implement the depth-from-size formula on
  `SNv3D.csv` (`ball_bbox` + `calibration` → 3D) and compare with `ball_3D`. Do you get
  ~4.2 m mean error too? This is the number your physics fit has to beat.
- [YOU] Clean 2D ball track (fill small gaps, remove false detections).
- [YOU] Kick frame: ball position on the ground via Stage 1 H.
- [YOU] Fit a 3D trajectory from the kick point (start velocity as unknowns, gravity
  fixed, drag optional) with `scipy.optimize.least_squares`, minimizing reprojection
  error against the 2D track. Needs the full camera (not just H) — see Stage 1 stretch
  or get it from the Colab calibration output.
- [YOU] Sanity checks: speed in km/h plausible? Height at the goal line plausible?
- [YOU] Compare on your clip: physics fit vs per-frame size-based depth. Where do they
  disagree most, and which do you trust there?

**Checkpoint**
- Ball trajectory reprojects onto the video track; speed and height are physically
  believable. You can explain what would make the fit wrong.
- Detection rate of the ball model on your clips (a number).
- Size-based baseline error on `SNv3D.csv` (a number, vs the paper's 4.2 m).

**Stretch — learn triangulation properly:**
1. First with KTH Football II: 3 synchronized views + 2D joints + cameras → triangulate
   the joints yourself, compare with their 3D ground truth. Their cameras are
   orthographic, so it's plain linear least squares — the gentle version.
2. Then perspective: two phones on tripods, a friend kicking a ball, clap to sync,
   checkerboard to calibrate, `cv2.triangulatePoints`. Compare two-camera result vs your
   single-camera physics fit, and a tape-measured distance.

---

## Later (parked — only after Stage 5)
- **Triangulate the real shot using a broadcast replay.** Broadcasts often replay the shot
  from behind the goal = a second view of the same ball. SoccerNet-v3D builds its ground
  truth exactly this way. The hard part: syncing a slow-motion replay to live frames, and
  calibrating a camera that only appears in the replay.

---

## Data formats (the contract between stages)

`tracks.json` = `data/tracks/<clip>.json`, written by `src/track/to_pitch.py`. `team` is `null` until
`src/track/teams.py` fills it in with `"A"` / `"B"` / `"other"`; `ball_px` is `null` in frames without
a ball. Each player also carries `box_px` ([x1, y1, x2, y2], the box the dot came from): `join_tracks`
renames ids, so the raw tracking file can no longer be matched by id, and teams/pose need the crop.
```json
{
  "clip": "clip01",
  "fps": 25,
  "frames": [
    {
      "frame": 0,
      "players": [{"id": 7, "team": "A", "x": -3.2, "y": 14.8, "visible": true}],
      "ball_px": [812.0, 403.5]
    }
  ]
}
```

`camera.json` = `data/camera/<clip>.json` — per frame: `frame`, homography `H` (3x3, pitch meters →
pixels, `H[2][2] = 1`), `pnl_rep_err_px` (PnLCalib's self-reported error, `null` if it found no
camera), `filled` (`true` = PnLCalib's camera was missing or broken, H was blended from the good frames
around it with `h_at`). Frames before the first / after the last good frame are left out. Written by
`src/calib/export_camera.py`. Full camera (K, R, t) gets added in Stage 5.

Raw tracking = `data/track/<clip>_<model name>_<tracker name>.json`, written by
`src/track/detect_track.py`. Pixels, straight from YOLO + tracker, before any filtering:
`{"clip", "model", "tracker", "frames": [{"frame": 0, "boxes": [{"id": 3, "cls": "player",
"conf": 0.91, "xyxy": [x1, y1, x2, y2]}]}]}`. `id` is only stable until an ID switch; a ball track
can carry a person `cls` for a few frames (`to_pitch.py` treats any ID that was ever "ball" as ball).

`data/pose/npz/pose_<id>.npz` — per frame: SMPL `global_orient` (3), `body_pose` (69), `betas` (10),
`frame` index.

`ball3d.json` — per frame: `x, y, z` in meters + fit residual.

`annotations/clicks/<clip>_<frame>.json` — hand-clicked pitch points for one frame:
`{"frame": "...", "points_px": {"box_front_right": [u, v], ...}}`. Names = keys of
`POINTS` in `src/calib/pitch_model.py`; points not visible are simply absent. Tracked in
git (hand-made, can't be regenerated).

Units: meters and pitch frame everywhere, except `*_px` fields (pixels).
External data in other frames (e.g. SoccerNet: origin at center spot) gets converted to
the pitch frame when loaded — never stored in its original frame.

---

## Resources (download when you reach the stage, not before)

| Resource | What's in it | Used in | Size | Notes |
|---|---|---|---|---|
| SoccerNet GSR (validation clip) | Broadcast clips + player positions in meters | 2 | whole split (check size) | Ground truth for position error. **Next download** |
| Roboflow `football-player-detection-v9.pt` ([roboflow/sports](https://github.com/roboflow/sports/tree/main/examples/soccer), link in `examples/soccer/setup.sh`) | YOLOv8x fine-tuned on Bundesliga broadcast: ball, goalkeeper, player, referee | 2 (in use) | 130 MB | **Downloaded** to `data/models/` (by me, by hand, from Google Drive id `17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q`). sha256 `75b09c37…fffaf`. AGPL (Ultralytics). Same repo has `football-ball-detection.pt` (ball, Stage 5 option) and `football-pitch-detection.pt` (pitch keypoints) |
| Ultralytics `yolo26m.pt` | General COCO detector ("person", "sports ball") | 2 (compared, not used) | 42 MB | Downloads itself into `data/models/` |
| [SoccerNet-v3D](https://github.com/mguti97/SoccerNet-v3D) — `SNv3D.csv` | Per image: full camera (K,R,t), 2D ball box, triangulated 3D ball | 1 stretch, 5 | 3.6 MB | No code in the repo, only data + weights (release v1.0.0). Paper: arXiv 2504.10106 |
| SoccerNet-v3D — `yolo-sn-ball-opt.pt` | YOLOv11 ball detector, fine-tuned on broadcast | 5 | 49 MB | GPL-2.0. `*.pt` is gitignored |
| [KTH Multiview Football II](https://www.csc.kth.se/cvap/cvg/?page=footballdataset2) — 3D part | 3 synced views, 800 frames, 14-joint 3D pose GT, camera per frame | 4, 5 stretch | ~200–250 MB per sequence | Academic use only. 2013, close-up footage. Download one sequence only |

KTH layout (checked on `data/kth/sequence2/`, 175 frames): text files with one number per line.
`positions3d.txt` → `reshape(F,14,3)`; `positions2d.txt` → `reshape(F,3,14,2)` (per frame, per
camera); `cameras.txt` → `reshape(F,3,4,2).transpose(0,1,3,2)` = 2×4 affine camera per frame
per camera (MATLAB column-major). Joints in LSP order: R ankle, R knee, R hip, L hip, L knee,
L ankle, R wrist, R elbow, R shoulder, L shoulder, L elbow, L wrist, neck, head top.
3D is body-centred, `z` up, ~meters. Images are 480×640 crops that follow the player.
GT quality: reprojection median 5.7 px, but the **left leg/wrist** is off by up to 100 px
(~0.45 m) in frames ~77–87 (occluded challenge) and a few others. Bone lengths vary
±4–6 cm between frames, so the GT itself is noisy by about that much.

---

## Things NOT to do (yet)
- Don't download Stage 4/5 datasets before you reach those stages.
- Don't start with 40-second clips. 5–10 s.
- Don't fight CUDA installs on the Mac. Kaggle (my choice) or Colab.
- Don't tune a fix (smoothing, joining, thresholds) by how the video looks alone — measure it.
- Don't add VR, apps, or the full goalkeeper product before Stage 3's decision gate.
- Don't trust any output you haven't reprojected onto the video.
