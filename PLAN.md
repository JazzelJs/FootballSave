# PLAN.md — Football clip to 3D

Tags: `[YOU]` I write it · `[TOGETHER]` shared · `[CLAUDE]` boilerplate Claude writes.
Each stage: **Learn** → **Build** → **Checkpoint** (must pass) → **Explain it back**.

Rule of thumb: if a stage takes more than ~2x the estimate, stop and ask Claude
"what am I overcomplicating?" before continuing.

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
- [CLAUDE] Run Ultralytics YOLO + ByteTrack locally with `device="mps"`, save raw boxes.
- [YOU] Foot point = bottom-center of each box → pitch meters using Stage 1 H.
- [YOU] Write `tracks.json` (format below).
- [CLAUDE] 2D minimap video (top-down pitch with dots) next to the original frame.
- [YOU] Download one SoccerNet Game State Reconstruction (GSR) validation clip (moved from
  Stage 0). Ground-truth player positions in meters, from another match and stadium.
  Check the download size first (the downloader fetches a whole split). Watch out: 25 fps,
  origin at the centre spot (convert to our frame), and a moving camera, so pick a steady
  5–10 s stretch or use the Stage 1 moving-camera method to get H per frame.
- [YOU] Evaluate on the SoccerNet GSR clip: match your players to ground truth and
  compute mean position error in meters. Look at the worst cases and say WHY they're bad.
- [TOGETHER, optional] Run the full sn-gamestate baseline in Colab on the same GSR clip
  and compare with your simpler pipeline. Where does the baseline win, and why?

**Checkpoint**
- Minimap video looks right.
- A measured number: mean position error on the GSR clip, e.g. "1.4 m, worst 4 m at
  frame 212 because of occlusion".

**Explain it back:** Where does position error come from? (Calibration? Box bottom?
Occlusion?) Which one is biggest in your data?

---

## Stage 3 — 3D capsule viewer + goalkeeper camera  (~2 sessions)

**Learn**
- Coordinate system conversion (pitch z-up → three.js y-up).
- Interpolating positions between frames; camera field of view.

**Build**
- [CLAUDE] three.js viewer skeleton in `src/viewer/`: pitch plane with lines, loads
  `tracks.json`, play/pause/scrub.
- [YOU] Player capsules (1.8 m tall), team colors, update each frame.
- [YOU] Two cameras: free orbit camera and a **goalkeeper camera** at goal center,
  eye height ~1.7 m, looking at the ball or shooter.
- [TOGETHER] Toggle to show which players were never visible in the broadcast frame
  (gaps in their tracks).

**Checkpoint**
- You can watch the clip in 3D from the goalkeeper position.

**Decision gate (be honest):** Is the goalkeeper view useful with just capsules? Which
missing information hurts most — body pose, the ball, or off-screen players? Use the
answer to choose between Stage 4 and Stage 5 next.

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
- [TOGETHER] Colab notebook: run a single-view model (e.g. GVHMR or 4DHumans) on crops
  of 1–3 key players (the shooter first). Needs SMPL model files (free registration).
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
  frame, your broadcast shooter is ~100–200 px tall.
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
teams are classified; `ball_px` is `null` in frames without a ball.
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
pixels, `H[2][2] = 1`), `pnl_rep_err_px` (PnLCalib's self-reported error). Frames without a camera are
left out. Written by `src/calib/export_camera.py`. Full camera (K, R, t) gets added in Stage 5.

`pose_<id>.npz` — per frame: SMPL `global_orient` (3), `body_pose` (69), `betas` (10),
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
| SoccerNet GSR (validation clip) | Broadcast clips + player positions in meters | 2 | whole split (check size) | Ground truth for position error |
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
- Don't fight CUDA installs on the Mac. Colab.
- Don't add VR, apps, or the full goalkeeper product before Stage 3's decision gate.
- Don't trust any output you haven't reprojected onto the video.
