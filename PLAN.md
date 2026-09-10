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
- [CLAUDE] Project setup with `uv`, `.gitignore`, folder structure, `requirements`.
- [CLAUDE] `ffmpeg` command/script to cut a clip and extract frames to `data/frames/<clip>/`.
- [YOU] Pick **3–5 clips**, each: one continuous camera shot (no cuts), 5–10 s, the
  penalty box clearly visible, a shot on goal at the end. Write in `data/clips/README.md`
  for each clip: start/end time, fps, resolution, how much the camera moves.
- [YOU] Download one SoccerNet Game State Reconstruction (GSR) validation clip. This has
  ground-truth player positions in meters — you'll use it to measure your errors.

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
- [YOU] `src/calib/homography.py`: compute H with `cv2.findHomography` from your
  clicked points.
- [YOU] Convert a clicked foot position (pixel) → pitch meters.
- [TOGETHER] Draw the pitch model lines back onto the frame using H (reprojection overlay).
- [TOGETHER] Warp the frame to a top-down view with `cv2.warpPerspective`.
- [YOU, stretch] Implement the homography yourself with DLT + SVD in NumPy and compare
  with OpenCV's result.

**Checkpoint**
- Reprojected pitch lines sit on the real lines. Mean reprojection error **< 5 px** on
  points you did NOT use to compute H (hold some out!).
- Top-down view: penalty box looks rectangular with correct proportions.

**Explain it back:** Why can't this homography tell you where the ball is when it's in
the air? What happens to accuracy if all your clicked points are close together?

**Moving camera (after the easy clip works):**
- [YOU] Calibrate every ~10th frame by hand, interpolate H in between. Measure how bad
  interpolation is on a frame you calibrated but held out.
- [TOGETHER] Then try an automatic method (PnLCalib or No-Bells-Just-Whistles) in Colab,
  and compare it with your manual calibration on the same frames.

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
- [CLAUDE, stretch] Smoothing filter; retarget to a rigged character in Blender.

**Checkpoint**
- Shooter's pose reprojects within a reasonable pixel error (report the number).
- You can state how wrong the facing direction is on average.

**Explain it back:** Why is position taken from the pitch and not the pose model?
Why is facing direction the error that matters most for a goalkeeper?

---

## Stage 5 — The ball in 3D  (~3 sessions)

**Learn**
- Why one camera can't locate a flying ball (the ray problem).
- Priors that help: kick point on the ground, gravity + drag, known ball size (~22 cm).
- Least-squares fitting; why the answer is only as good as the priors.

**Build**
- [CLAUDE] Ball detection. Start with YOLO "sports ball"; if it misses too often,
  fine-tune on the SoccerNet v3 H250 dataset (this is where that repo is useful).
- [YOU] Clean 2D ball track (fill small gaps, remove false detections).
- [YOU] Kick frame: ball position on the ground via Stage 1 H.
- [YOU] Fit a 3D trajectory from the kick point (start velocity as unknowns, gravity
  fixed, drag optional) with `scipy.optimize.least_squares`, minimizing reprojection
  error against the 2D track. Needs the full camera (not just H) — see Stage 1 stretch
  or get it from the Colab calibration output.
- [YOU] Sanity checks: speed in km/h plausible? Height at the goal line plausible?

**Checkpoint**
- Ball trajectory reprojects onto the video track; speed and height are physically
  believable. You can explain what would make the fit wrong.

**Stretch — learn triangulation properly:** two phones on tripods, a friend kicking a
ball, clap to sync, checkerboard to calibrate, `cv2.triangulatePoints`. Compare
two-camera result vs your single-camera physics fit, and a tape-measured distance.

---

## Data formats (the contract between stages)

`tracks.json`
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

`camera.json` — per frame: homography `H` (3x3), later full camera (K, R, t) if available.

`pose_<id>.npz` — per frame: SMPL `global_orient` (3), `body_pose` (69), `betas` (10),
`frame` index.

`ball3d.json` — per frame: `x, y, z` in meters + fit residual.

Units: meters and pitch frame everywhere, except `*_px` fields (pixels).

---

## Things NOT to do (yet)
- Don't start with 40-second clips. 5–10 s.
- Don't fight CUDA installs on the Mac. Colab.
- Don't add VR, apps, or the full goalkeeper product before Stage 3's decision gate.
- Don't trust any output you haven't reprojected onto the video.
