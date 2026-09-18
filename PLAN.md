# PLAN.md — Football clip to 3D

Tags: `[YOU]` I write it · `[TOGETHER]` shared · `[CLAUDE]` boilerplate Claude writes.
Each stage: **Learn** → **Build** → **Checkpoint** (must pass) → **Explain it back**.

Rule of thumb: if a stage takes more than ~2x the estimate, stop and ask Claude
"what am I overcomplicating?" before continuing.

---

## Current status — READ THIS FIRST (updated 2026-09-18)

**Stages 0–5 are all complete and measured. There is no blocked work.** What is left is mine to
write, not to build: the "Explain it back" answers and the `LEARNING_LOG.md` entries.

**The pipeline end to end, and what each stage is worth:**

| stage | what it does | the number |
|---|---|---|
| 1 camera | PnLCalib → H and the full camera per frame | 0.55 m median on the grass |
| 2 players | YOLO + BoT-SORT → positions, teams | **0.48 m** median, 89% team correct |
| 4 bodies | HMR2 → SMPL, rotated into pitch space | **113 mm** PA-MPJPE, facing 6–22° |
| 5 ball | one-camera physics fit | **3.70 px**, 117 km/h, inside the post |

**Two clips are demo-ready. `SHOWCASE.md` has the links, the talking points and the limits.**

| | SNGS-043 (a goal) | clip04 (a near miss, our clip) |
|---|---|---|
| ground truth | yes, SoccerNet labels | **none at all** |
| posed bodies | 4 (tracks 1131, 284, 17, 171) | 4 (tracks 10, 15, 17, 9) |
| ball in 3D | 117 km/h, 3.70 px, **inside** ✅ | 106 km/h, 9.20 px, **0.87 m outside** ✅ |
| best view | `?clip=SNGS-043`, 24.3 s, press `c` | `?clip=clip04`, 4.2 s, `pose view` |

Both fits land on the one fact nobody annotated — the goal scored, the near miss missed.

**If you change anything, these must still pass:**
```
uv run python src/ball/clean.py --self-check
uv run python src/ball/fit.py --self-check          # 3.70 px, 117 km/h, x +3.24 z -0.07
uv run python src/ball/size_baseline.py --self-check
uv run python src/pose/orientation_error.py SNGS-043 284   # 10.4° / 66.6°, unchanged since 09-18
```

**What to read, in order, and nothing else:** this block → "Next session" below → the runbook →
`SHOWCASE.md` if you are demoing → `git log --oneline -10`. **The dated history below is
reference, not required reading** — go there only when you need to know why a decision was made.

**SNGS-028 is the third fitted clip (2026-09-18), and getting there corrected a wrong call.**
`KICK["SNGS-028"] = 387`, `WINDOW_END["SNGS-028"] = 395`. Result: **8.57 px** over 9 frames,
**78 km/h** at 17.7°, crossing the goal line at **x −4.16 m, z 2.20 m** — just outside the post and
just under the bar, which is what `action_class` "Shots off target" should look like.
- **The first attempt failed at 24.79 px and I wrongly blamed the ball being airborne.** The test I
  used — comparing the pixel motion a grass-bound ball would make against the motion it does make —
  is **confounded by camera pan**: a broadcast camera follows the ball, so a moving ball can sit
  almost still in the image. Low pixel motion with high grass motion is what *tracking* looks like,
  not what *flying* looks like. That test cannot tell the two apart and should not be used.
- **The real fault was the automatic flight window.** `flight_window()` ends the flight where the
  pixel track reverses, which is the ball hitting the net. **A shot off target never reverses**, so
  it ran to the end of the data (387–413) and the fit degraded 8.57 → 24.79 px as the window grew.
- **The late frames are bad annotation, not bad physics.** From frame 396 the steps alternate
  64, 7, 37, 74, 2, 71 — the mistimed-label signature from A2, but with steps too large for
  `clean.py`'s ratio test to catch — and the box stays a constant **20–21 px** while the ball flies
  40 m away, where SNGS-043's shrank 24 → 11 px. The annotator stopped tracking the real ball.
- **Also found, and reusable:** SoccerNet's labels carry `info.action_position` and
  `info.clip_start`. For SNGS-043 that gives frame **618.7**, and our fit puts the ball crossing the
  goal line at **617.4** — **1.3 frames (52 ms) apart**, an independent check on Stage 5 from an
  annotation we had never opened. It marks the *event*, not the kick: on SNGS-043 (Goal) it is the
  ball crossing the line, on SNGS-028 (Shots off target) it is the strike itself.
- **Still weak here:** 9 frames against 6 unknowns is thin, and our H disagrees with the labels by
  5.04 m at the kick point (1.55 m on SNGS-043). Treat 78 km/h as the least trustworthy of the
  three speeds.

**Three traps this project has already fallen into.** They will recur:
1. **A track id means nothing without its clip.** SNGS-043 and clip04 both have a track 17.
   Pose files are `pose_<clip>_<track>.npz`; the viewer's `ALL_POSES` carries a `clip` field.
2. **Per-pixel rules break when the camera pans.** "Objects that never move" found zero in pixel
   space and the culprit in metre space. Ask whether a rule belongs in the image or on the grass.
3. **Never tune a parameter against a metric it can game.** The yaw-smoothing sweep falls
   monotonically to a 0.8 s window, because the travel-direction reference is itself smooth.

---

### Where we are, in detail


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

**Start the next session with:**
1. **Stage 4 checkpoint is complete** (19.7 px reprojection · 113.3 mm PA-MPJPE · facing stated both
   ways). Owed before it closes, both mine: the three "Explain it back" answers, and the
   `LEARNING_LOG.md` wrap-up for 2026-09-15.
2. **Stage 5 is COMPLETE (2026-09-18): A, B, C, D, E all done and measured.**
   All of it was written by Claude on my "just do it" / "do it for me" under time pressure, so the
   whole stage owes me the walk-through and the "Explain it back".
   Headline: the shot is fitted in 3D from one camera at **3.70 px median reprojection**,
   **117 km/h**, crossing the goal line **0.42 m inside the post** — a goal, which is free ground
   truth nobody annotated. Pipeline: `from_labels.py` → `clean.py` → `fit.py` → `draw.py`, and
   `detect.py --write` in place of labels on a clip that has none.
   **Nothing in Stage 5 is blocked. What is owed is all mine:** the walk-through I asked Claude
   to defer, the "Explain it back", and the `LEARNING_LOG.md` entries.
   **Both clips, side by side — and each one's checkable fact came out right:**
   | | SNGS-043 (a goal) | clip04 (a near miss) |
   |---|---|---|
   | kick frame | 602 | 196 |
   | usable frames | 15 | 29 |
   | reprojection | **3.70 px** | **9.20 px** |
   | launch speed | 117 km/h | 106 km/h |
   | at the goal line | x +3.24 m → **inside** ✅ | x +4.53 m → **outside** ✅ |
   **The four things in this stage I should be able to explain before it closes:**
   - Why the unconstrained fit chose 272 km/h *into the ground*, and why `vz ≥ 0` is a
     measurement-free fact that fixes it.
   - Why two answers 0.4 px apart can be 70 km/h and 171 km/h, and what that says about how much
     a single camera can ever know.
   - Why free flight ends at the goal line (617) and not where the pixel step collapses (622).
   - Why binning "objects that never move" in pixels found nothing, and in metres found the
     culprit — and what that says about every other per-pixel rule in this repo.
   - Why depth-from-size is wrong by a fixed *percentage* of the distance rather than a fixed
     number of metres, and why a correction factor could not fix it.
   - Why frame 601 is a *corner* and frame 204 is an *outlier*, when both sit off the line through
     their neighbours — and why no threshold can tell them apart cleanly.
   **Known soft spots, stated rather than hidden:** k = 0.046 1/m is 3.5× a real ball's, so drag
   is absorbing other errors; C2's 0.42 m margin is smaller than our camera's 2.25 m error at the
   ball; C3's shadow plot is dominated by that same camera offset, so it cannot measure height on
   this clip; the ball detector's whole-clip recall is only 53% even at native resolution; and
   PLAN.md's 4.2 m figure for depth-from-size does not survive contact with the data (we measure
   17.9 m), so the baseline our fit "had to beat" was set four times too kindly.
3. **clip04 poses DONE 2026-09-18** — I ran the Kaggle notebook myself
   (`notebooks/kaggle/football3d_clip04_poses.ipynb`), 4 tracks, 337/313/332/337 frames at 49.95 fps.
   **clip04's poses beat SNGS-043's, and exactly where predicted — in the tails:**
   | facing error, median / 95th | track 10 (shooter) | 15 | 17 | 9 |
   |---|---|---|---|---|
   | samples above 3 m/s | 61 of 337 | 128 of 313 | 86 of 332 | 62 of 337 |
   | **pitch space** | **11.4° / 23.2°** | 22.3° / 51.8° | 15.7° / 42.2° | **6.3° / 23.9°** |
   | camera space, best-case zero | 7.4° / 139.1° | 41.1° / 134.2° | 8.5° / 87.6° | 9.2° / 45.8° |
   SNGS-043's were 10.4°/66.6°, 15.2°/52.0°, 29.9°/104.1°. Medians are comparable; **95th
   percentiles are 2–4× better**, because clip04's tracks are unbroken and SNGS-043's worst frames
   were always join hand-overs. Boxes 86–98 px vs 78. Unlike SNGS-043 the shooter has a real
   number here (61 running samples, not 4).
   **Honest oddity:** on tracks 10 and 17 the camera-space median is *lower* than pitch space
   (7.4 vs 11.4, 8.5 vs 15.7). That row gets the offset minimising its own error and clip04's
   players run in a narrow spread of directions, so one offset fits the middle. Its tails give it
   away: 139.1° vs 23.2°.
   **Pose files are now named per clip.** A track id means nothing without its clip: SNGS-043 has
   a track 17 and so does clip04, and downloading the new one nearly destroyed the old — the
   browser saved it as `pose_17-3.npz` instead of overwriting, which is the only reason nothing
   was lost. Three scripts built the flat `pose_{id}.npz` path, so the collision was in the code
   too. New `orient.pose_npz(clip, track)` / `orient.pose_mesh(clip, track)` →
   `pose_<clip>_<track>.npz`; `draw_facing.py`, `orientation_error.py` and `reproject_smpl.py` use
   it, and the 8 active files were renamed. SNGS-043's numbers are unchanged after the rename
   (track 284 still 10.4° / 66.6°), which is how I know nothing else moved.
   **`--smooth-yaw` swept 1..15 (and out to 41), 2026-09-18 — and the sweep cannot choose it.**
   | median facing error | w=1 | w=5 | w=9 | w=15 | w=31 | w=41 |
   |---|---|---|---|---|---|---|
   | clip04 track 17 | 16.7° | 15.7° | 15.7° | 13.9° | 11.3° | 11.6° |
   | clip04 track 15 | 21.8° | 22.3° | 21.0° | 20.2° | 19.2° | 18.4° |
   | SNGS-043 track 284 | 11.2° | 10.4° | 10.3° | 9.0° | 8.6° | 8.2° |
   It falls **monotonically and never turns around**, so "optimise the window" means "smooth until
   the signal is gone" — and w=41 is 0.82 s on clip04, 1.64 s on SNGS-043, long enough to erase
   real turning (a player turns 180° in about half a second). **The proxy prefers smoothing because
   the proxy is itself smooth:** travel direction comes from tracker positions 4 frames apart, so
   it rewards anything that makes the pose yaw smoother whether or not the pose got better. A
   parameter must never be tuned against a metric it can game.
   What the sweep *does* establish: across w=1..15 the median moves by only **0.8–3.3°** on every
   track of both clips — inside the proxy's noise. So pick the window on physics and keep it
   consistent in **time**: raw per-frame yaw change is 1.5–2.2°/frame on clip04 against 3.6–4.2°
   on SNGS-043, a ratio of ~2 which is exactly 49.95/25 — the same angular noise spread over twice
   as many frames. SNGS-043's w=5 at 25 fps is 0.20 s, so clip04 is re-exported at **w=9**
   (0.18 s). The numbers did not move: 12.0 / 21.0 / 15.7 / 6.4° against 11.4 / 22.3 / 15.7 / 6.3°.
   **The step I had left out of my own instructions:** HMR2 saves pose *parameters*, not vertices,
   so `apply_smpl.py --input … --output …_vertices.npz` runs **before** `export_smpl_mesh.py`.
   `export_kaggle_inputs.py` now prints the full seven steps.
4. **Viewer fixes, 2026-09-18** (found by asking which clips are showcasable):
   - The 2D fallback ball read `ball_px` from `tracks.json` — the *player* detector's ball, which
     on clip04 jumps **1690 px** to the spare balls by the ad boards. It now reads
     `data/ball/<clip>_clean.json` (`label` rows only), falling back to `tracks.json` only for a
     clip with no ball file (SNGS-028).
   - **Poses were not clip-aware.** `POSES` was a flat list of SNGS-043's four tracks loaded on
     every clip, so clip04 announced "4 posed tracks" it does not have and id 1131 could have
     landed on whoever clip04 calls 1131. Now `ALL_POSES` carries a `clip` field and `POSES`
     filters on it; clip04 reads "capsules only, no poses". An SNGS-043-only `console.assert`
     (frame 599) is guarded too.
5. **`SHOWCASE.md` is the two demo versions** — what each clip has, the numbers, the exact URL and
   frame to scrub to, and the honest limits to admit if asked.
6. **Parked, not blocking:** track 171's facing is 30° median over 68 running samples and its worst
   frames (568–572) are the hand-over inside the joined track. That is a tracking problem, and it is
   the Stage 2 team-colour idea coming back.
7. **Wrap-up still owed for 2026-09-15** (CLAUDE.md rule 5): detection vs tracking, ID switches, NMS,
   *where* vs *who*, foot point → meters, wobble, goal side. 2026-09-16 has its `LEARNING_LOG.md`
   entry; parts of it are Claude's wording and I should rewrite those in my own words.
8. Open questions I haven't answered yet (no rush, they're small):
   - (B)'s biggest error is exactly 3.00 m while (A)'s is 12.9 m. Why, and what does that do to
     comparing (B) with (A)?
   - If PnLCalib gave a perfect camera tomorrow, which file changes: `data/track/…` or `data/tracks/…`?
9. Left in Stage 2 on purpose, to pick up when it matters: use the team colours to refuse a join or a
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


---

## History — how each number was reached (reference, not required reading)

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

**Execution order (written 2026-09-18, so a new session can just pick the next unticked box).**
Nothing in A–C needs a download. Measured facts to start from: the SNGS-043 labels carry the ball
in **738 of 750 frames**, missing 367 and 476–481 and 558–562; the box is **3–25 px wide, median
12**; the kick is **frame 602**; the focal length is 3557 px at 602 and 4164 px at 620.

**A. The 2D ball track, from the labels (no download).**
- [x] A1 [CLAUDE] **done 2026-09-18**: `uv run python src/ball/from_labels.py SNGS-043` →
  `data/ball/SNGS-043.json`. Ball in 738/750 frames; gaps 367, 476–481, 558–562; box 3–25 px wide,
  median 12. Each row keeps `px` (a real measurement, the only input to a fit) and `pitch_px` (where
  a ray through that pixel hits the grass — the ball's *shadow* once it is airborne). Self-check:
  `--self-check` asserts the shadow is inside the pitch at 601 and past the goal line at 620.
- [x] A2 [CLAUDE on my "just do it", walk-through owed to me] **done 2026-09-18**:
  `uv run python src/ball/clean.py SNGS-043` → `data/ball/SNGS-043_clean.json`, same rows plus
  `source` = `label` / `filled` / `dropped`. **Only `label` rows may feed the B fit.**
  Result: 636 label, **1 filled**, 102 dropped; **18 of the 21 flight frames usable**.
  - **Two of the three gaps are refused, not filled.** One rule: compare the ball's direction just
    before the gap with the straight line across it. 367 agrees (dot 1.00) → filled. 476–481
    (dot −0.95) and 558–562 (dot −0.99) *reverse*: over 555–557 the ball goes right and comes back
    144 px left, so a straight line there is an invention, not a guess. Both are pre-kick, so
    refusing costs the fit nothing — that is the real finding: **gap filling changes no Stage 5 number.**
  - **The frames that do matter are 616, 618, 621** (steps 3.0 / 4.2 / 3.6 px where the ball covers
    15–25). Each is followed by a double step, so the annotation lagged a frame and caught up —
    mistimed, not stationary. Dropped by a ratio against the median of the **five steps before**
    the frame (`RATIO` = 0.35, `MIN_MOVE` = 1.0 px, `WINDOW` = 5). Past-only matters: a window that
    also looks forward borrows the post-622 in-net steps and lets 621 survive. Ratios 0.15 / 0.26 /
    0.24 vs 0.59–2.91 for every other flight frame. Subsumes the 41 exact pixel repeats.
  - **The 14 boxes ≤ 4 px wide are NOT dropped**: all at frames 29–134, none inside the flight, so a
    size rule changes no number here. Telling the match ball from a spare by the ad boards needs
    somebody to look at 14 crops.
  - **A mistake worth keeping:** the first rule dropped only *exact* pixel repeats and the self-check
    still passed — it counted survivors (21 ≥ 15), and a rule that drops nothing also leaves 21
    standing. The check now asserts the dropped set **is exactly** {616, 618, 621}.
- [x] A3 **done 2026-09-18** (`src/ball/fit.py`, `kick_frame` + `kick_point`). Two findings, both bad
  news that had to be faced:
  - **The kick frame cannot be detected, it is a per-clip fact.** `KICK = {"SNGS-043": 602}`.
    The pixel step does shout at the kick (2.2 px → 48.3 px) but the biggest step in the clip is
    frame **375 at 81.6 px** — a pass — and the ratio against the local median cannot separate them
    either (602 is 10× its neighbourhood, 375 is 9×). A shot is not "the fastest the ball moves";
    it is the event that ends in the net, which no local rule can see. Unknown clip → the script
    prints the ten biggest steps and asks for `--kick`.
  - **The check FAILED as the plan wrote it.** 602 vs 601 through our H moves **1.67 m**, not "a few
    tens of cm". But that mixes two causes, so it was measured properly instead: at frame 601 the
    ball is definitely on the grass, and our H vs the labels' own calibration on that same ball is
    **2.25 m apart** — against the 0.55 m median we measured on players' feet in Stage 2. The ball
    sits near the goal at the edge of the frame, where our H is least constrained. **This is the
    biggest single error in the stage**, and it is why p0 became an unknown (see B2).

**Measured on 2026-09-18, before starting A2 — use these instead of re-deriving them:**
- **Every frame of the flight has its own PnLCalib camera.** 0 filled frames in 595–640, so the
  nearest-trusted-camera dance that `src/pose/orient.py` needs does not apply to the ball fit.
- **The kick is frame 602 in one number:** the ball's pixel step goes 2.2 px (601) → **48.3 px** (602).
- **Free flight ends at about frame 622.** The pixel step collapses (622→623 is 1.0 px, 625→626 is
  0.0 px) and the track *reverses*: y climbs 448 → 495 over frames 622–640. That is the ball in the
  net, not in the air. **So the fit window is roughly 602–622, about 21 frames / 0.84 s.** Fitting
  past 622 will drag a parabola through a ball that has already stopped.
- The box shrinks 24 px → 11 px across the flight (it is moving away), then sits at 12–13 px in the net.
- Frames with a suspiciously small step mid-flight — 616 (3.0 px), 618 (4.2 px), 621 (3.6 px) — are
  label jitter, probably a repeated annotation. **A2 should decide what to do with them**; they are
  not the ball standing still at 100 km/h.

**B. The physics fit (no download).** All done 2026-09-18 in `src/ball/fit.py`.
**Result: v0 = (+1.20, −32.36, +3.32) m/s, drag k = 0.0459 1/m, reprojection 3.70 px median /
4.62 mean / 11.73 max over 15 frames.**
- [x] B1 `path()`: closed form `p0 + v0·t + ½·g·t²` for k = 0, RK-free small-step integration for
  k > 0 (drag has no closed form). Self-checked against the schoolbook formula at t = 0 and 0.5 s.
- [x] B2 `residuals()` + `fit()`. **Three things had to be added before the fit meant anything:**
  1. **The unconstrained fit cheats.** Left free it returns v0 = (+45, −60, −7) m/s at 9.7 px — a
     **272 km/h** ball kicked *into the ground*, 8.5 m wide of a goal that was scored. It wins
     because an underground ball is no longer constrained by gravity at all. Fixed with physics,
     not tuning: `vz ≥ 0` (a ball on the grass cannot be kicked down through it), `|v| ≤ 45 m/s`,
     and a penalty on any negative z along the path.
  2. **p0 is an observation, not a constant.** Our H says y = 14.18 m, the labels say 12.74 m, and
     our camera is worth ~30 px at the ball's 65 m — wider than the whole residual. Nailed to
     either number the velocity just absorbs the difference. So p0 drifts on a 1.5 m leash
     (`P0_SIGMA`), z0 stays exactly 0. Final drift: 0.1 m.
  3. **The window from A2 was too long.** The pixel rule said 602–622, but the first fit crosses
     the goal line at frame **617.4** — so 619–622 are the ball already IN the net, decelerating
     against it, and fitting them as free flight bent the parabola 0.31 m under the pitch. One
     refit on 602–617: **10.98 px → 3.70 px**. Free flight ends at the goal line, not where the
     pixel step collapses.
  Multi-start (15 guesses) because the residual is not convex: near/slow and far/fast look alike
  down one camera ray. **The ray problem, measured: two solutions 0.4 px apart — 70 km/h into the
  goal and 171 km/h 8.5 m wide of it.** Pixels cannot separate them. Gravity cannot either,
  because a nearly flat shot has almost no parabola. Only "the ball is not underground" can.
- [x] B3 **Drag kept, and it is not marginal.** Same window, same everything else:
  | | no drag | with drag |
  |---|---|---|
  | reprojection median / max | 6.31 / 24.08 px | **3.70 / 11.73 px** |
  | launch speed | 109 km/h | 117 km/h |
  | at the goal line | x +6.04 m → **misses ❌** | x +3.24 m → **in ✅** |
  Without drag the model overshoots in the image and the error grows monotonically to the end of
  the flight. **Caveat:** k = 0.046 1/m is 3.5× a real football's ~0.013 and sits near its bound,
  so it is absorbing something else too (camera error, spin, the flat-shot ambiguity). Honest
  reading: drag-shaped deceleration is real and needed; the *value* of k is not a measurement.

**C. Checks that need no ground truth (no download).** All done 2026-09-18.
- [x] C1 **117 km/h** (32.6 m/s), rising 5.9° off the grass. A hard shot is 90–120 km/h, so this
  passes — and it is the check that caught the 272 km/h cheat above. It earned its place.
- [x] C2 **At the goal line: x = +3.24 m, z = −0.07 m → INSIDE the posts (±3.66 m) ✅.** The shot
  crosses 0.42 m inside the keeper's-right post, along the ground. Two honest caveats: (a) 0.42 m
  of margin is *less* than our camera's own 2.25 m error at that spot, so this is "consistent with
  a goal", not proof of one; (b) my first version of this check demanded `z > 0` and reported a
  real goal as a miss — a ball rolling over the line *is* a goal, so the lower bound belongs in
  the physics (the residual), not in the definition of scoring.
- [x] C3 **Plotted, and it does NOT show what the plan predicted** —
  `outputs/ball_shadow_SNGS-043.png`. The gap between our fitted ground track and the labels'
  shadow is **1.55 m at the kick**, where the ball is ON the grass and the true gap is zero. It
  then dips to 0.26 m and grows to 1.0 m. So the plot is dominated by the constant ~2 m camera
  disagreement from A3, not by height: our fitted height only ever reaches 0.43 m, and a 0.43 m
  ball cannot make a 1.5 m shadow gap. **C3 cannot measure height on this clip** — the systematic
  error is bigger than the signal. It would work on a clip where our H and theirs agree better.
- [x] C4 `src/ball/draw.py` → `outputs/ball_fit_SNGS-043.mp4` (32 frames at 8 fps) and
  `outputs/ball_fit_SNGS-043_000602.jpg` / `_000612.jpg`. Red = the label's ball, green = the fit
  through that frame's own camera with its px error, yellow = the whole flight, grey = its ground
  shadow, so the gap between yellow and grey **is** the height. At frame 612 the two circles sit
  on top of each other (10.1 px, z = 0.61 m) and the curve goes into the net past the keeper.

**D. The numbers the checkpoint asks for (these need the two downloads in Resources).**
**Both downloads are in** (Claude fetched them on my "can you do it for me", 2026-09-18):
`data/models/yolo-sn-ball-opt.pt` (49 MB, GPL-2.0, one class `ball`) and `data/snv3d/SNv3D.csv`
(3.6 MB, 4051 rows). Beware the release also holds `yolo-sn-ball.pt` (same size, un-tuned) and
`yolo-issia-ball-opt.pt` (153 MB, other dataset) — we want the `-opt` SN one.
- [x] D1 **done 2026-09-18** (`src/ball/size_baseline.py`; Claude wrote it on my "do it for me",
  walk-through owed to me). **The answer is not the one this plan assumed.**
  | depth-from-size on SNv3D.csv | mean | median | as % of the distance |
  |---|---|---|---|
  | train (3241 rows) | 17.57 m | 14.14 m | 20.2% |
  | **test (810 rows, held out)** | **17.88 m** | **14.89 m** | **20.8%** |
  **~4× worse than the 4.2 m this plan credits to "this per-frame method".** The error is a fixed
  *share* of the distance in every band (19–22% from 25 m to 1000 m), which is the signature of a
  scale error, not noise. Two causes, measured:
  1. **The boxes are 23% bigger than the ball.** The geometry demands a median 11.99 px diameter
     at these distances; the annotations give 15.50 px. A generous box means a too-near ball, and
     `estimate/true` is 0.813 on median — biased low, exactly as that predicts.
  2. **Correcting the bias barely helps.** One factor fitted on train (k = 1.232), applied to
     test: mean 17.88 → 16.55 m, median 14.89 → 12.20 m. So the *per-row scatter* is the real
     problem, not the bias. With the ball 12–15 px wide, 1.5 px of annotation slop is ~12% of the
     diameter and therefore ~12% of the distance. This method cannot be rescued by calibration.
  **So the 4.2 m is probably not this formula, and I should stop quoting it as such.** The csv also
  carries `optimized_d`, which is **not** a camera-to-ball distance (17.1 m where the true distance
  is 77.3 m, and the ratio is not constant), so the paper is reporting some refined quantity.
  **Two format traps, both settled by measurement rather than assumption** (and both of Claude's
  first guesses in the skeleton were wrong, which is worth remembering):
  - `ball_bbox` (x, y) is the **centre**, not the top-left: projecting `ball_3D` through the
    calibration lands **2.77 px** from the centre and 12.94 px from the corner. That same check
    also proves `ball_3D` and `calibration` share one coordinate frame.
  - The skeleton claimed w and h differ a lot on a round ball (from row 0, 26.76 × 16.80). Across
    all 4051 rows **w ≈ h** (median 17.26 vs 16.73, ratio 1.02) — row 0 was an outlier. `min(w, h)`
    is still the right pick, because inflation can only ever make a box *bigger* than the ball:
    measured test mean error is min 17.9 m, height 20.4, width 21.5, max(w, h) 24.3.
  - `calibration` is a **Python** dict (single quotes → `ast.literal_eval`), with exactly
    PnLCalib's keys, so `camera_matrix()` takes it unchanged. `x_focal_length == y_focal_length`
    in 100% of rows, so the choice of focal is moot.
- [x] **"Compare on your clip: physics fit vs per-frame size-based depth"** (the last Build item),
  in the same script. Over the 16 flight frames of SNGS-043 the two methods **disagree by 6.0 m
  mean / 6.9 m median**. Depth-from-size has the ball receding 55.9 → 91.4 m; the fit says
  65.7 → 82.0 m. Size-based is biased *low* early (55.9 vs 65.7 at the kick) exactly as the 23%
  box inflation predicts, and it jitters ±10 m frame to frame because every frame is an
  independent guess off an 11–25 px box. **Which to trust: the fit** — not because it is prettier,
  but because it is checkable (3.70 px reprojection, and it puts the ball inside the posts) while
  depth-from-size has no way to be wrong *quietly*. Neither is ground truth: SNGS-043 has no 3D
  ball truth, so this measures disagreement, not accuracy.
- [x] D2 **done 2026-09-18, `src/ball/detect.py`.** Detection rate on SNGS-043, where the labels say
  exactly which 738 frames hold a ball. `conf` 0.10 deliberately low, a hit is within 25 px.
  | imgsz | ball found | centre error (median) | false alarms /12 | time |
  |---|---|---|---|---|
  | 640 (default) | **3.9%** | 3.6 px | 0 | 22 s |
  | 1280 | **43.9%** | 3.3 px | 7 | 70 s |
  | 1920 (native) | **53.1%** | 3.1 px | 9 | 158 s |
  **The default imgsz is the whole story.** The ball is 3–25 px wide (median 12) in a 1920-wide
  frame, and YOLO letterboxes to imgsz first, so at 640 a 12 px ball arrives as **4 px** — smaller
  than one stride-8 output cell, and the rate collapses to 3.9%. Native resolution costs 7× the
  time and buys 14× the recall.
  **The whole-clip 53% understates it for our purpose: during the flight (602–622) it finds the
  ball in 17 of 21 frames = 81%**, at ~3 px. A ball in the air against grass is easy; a ball at
  someone's feet is not. The paper's warning about poor transfer holds for the clip average, not
  for the shot itself.
  Honest caveat: the most confident box is a *different* ball in 30 of 392 hits, and 9 of the 12
  genuinely ball-free frames get a detection anyway.

**E. Our own clip, last.**
- [x] E1 **done 2026-09-18** (Claude did this one too, on my "do e1 for me"). `from_tracks.py` is
  **deleted** — reading "ball" boxes out of the *player* detector gave a track that was not the
  ball (pixel steps of 1690 px). Replaced by `src/ball/detect.py --write`.
  **Result on clip04, which has no ground truth of any kind: 106 km/h, rising 6.0°, reprojection
  9.20 px median over 29 frames, crossing the goal line at x = +4.53 m — 0.87 m OUTSIDE the post.
  clip04 is the near miss this project started from, so the fit agrees with the video** (C2 now
  reads `SCORED[clip]` and says "agrees" / "DISAGREES" instead of assuming every shot went in).
  `outputs/ball_fit_clip04.mp4`, `outputs/ball_fit_clip04_000210.jpg`. No C3 plot: a detector
  track has no labelled ground shadow to compare against.
  **The ball-candidate filters — only the third worked, and the two failures are the lesson:**
  | filter | result |
  |---|---|
  | most confident box | 316/337 frames, but 95% pixel step **608 px**, max 1293 |
  | + drop anything off the pitch (inv(H), 3 m margin) | 315/337, step **unchanged** |
  | + drop objects that never move, **binned in pixels** | **0 objects found** |
  | + drop objects that never move, **binned in metres** | 297/337, 95% step **31 px** |
  The culprit was one ball-shaped object at grass **(20.0, −2.1 m)**, just behind the goal line,
  which at frame 182 outscored the real ball 0.39 to 0.38. The on-pitch test missed it because
  −2.1 m is inside the margin the real ball needs. **The pixel-binned "static" test found nothing
  because the camera pans: a world-static object slides across the frame, so in pixels nothing is
  ever still.** Through `inv(H)` it sits in the same square metre all clip — 142 detections dropped.
  **Finding clip04's kick, since no rule can detect one** (SNGS-043 taught us that): the ball's
  *grass* track gives it away. It sits at (7.0, 16.5 m) barely moving over frames 189–192
  (steps 0.6–6.5 px), then runs 24 px/frame at 196 and crosses the goal line around 229. Contact
  is inside the **detection gap at 193–195** — and the gap itself is the signal, because a ball
  being struck is motion-blurred and the detector loses it. The frames confirm it
  (`outputs/clip04_kick_candidates.jpg`: the striker's boot meets it at 194–195, a blur streak by
  196). `KICK["clip04"] = 196`, the first frame where the ball is actually *measured* after contact.
  **A second cleaning rule was needed, and it exposed a bug of mine** (`clean.py`). clip04's frame
  204 jumps 235 px next to a missing frame 203, and the mistimed-label rule could not see it,
  because that rule only compares *consecutive* frames — which cut the flight window from 32
  frames to **7**. New rule: drop a row the straight line through its neighbours misses by more
  than `OUTLIER` local steps. **First try at `OUTLIER` = 3 threw away SNGS-043's frame 601** — the
  frame A3's whole camera check rests on — because 601 is the last frame before the boot hits the
  ball, so the track has a genuine **corner** there and the line from 600 to 602 cuts it. A corner
  looks exactly like an outlier to this test. Measured: 204 is 9.7× its local step off the line,
  601 is 4.9×, so `OUTLIER` = 6 sits between them. **That is the one weakly-determined number in
  the stage — it is not a measurement, and a sharper corner would need it revisited.**
- [x] E2 **done 2026-09-18.** `fit.py` writes `path` + `path_frames` (one (x, y, z) per frame) so
  the viewer never integrates drag in JavaScript. `src/viewer/index.html` now has two ball
  sources and shows which is which: **yellow** = a real 3D position with height, inside the fitted
  window; **white** = the old pixel-on-the-grass fallback, which is only true while the ball is
  actually on the grass. Verified in the browser: the header reads "ball in 3D for frames 602–617
  (117 km/h, 3.7 px)" and the goalkeeper view shows the ball low and to the keeper's right,
  matching C2's x = +3.24 m.

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
| [SoccerNet-v3D](https://github.com/mguti97/SoccerNet-v3D) — `SNv3D.csv` | Per image: full camera (K,R,t), 2D ball box, triangulated 3D ball | 1 stretch, 5 | 3.6 MB | **Downloaded 2026-09-18** to `data/snv3d/`. 4051 rows, all `is_ball` True, images 1280x720. `calibration` has exactly PnLCalib's keys, so `camera_matrix()` takes it unchanged. sha256 `bc1847d1…eb96e`. | No code in the repo, only data + weights (release v1.0.0). Paper: arXiv 2504.10106 |
| SoccerNet-v3D — `yolo-sn-ball-opt.pt` | YOLOv11 ball detector, fine-tuned on broadcast | 5 | 49 MB | **Downloaded 2026-09-18** to `data/models/`. GPL-2.0, `*.pt` gitignored. One class, `ball`. sha256 `a3082fb4…64f36`. Beware the release also has `yolo-sn-ball.pt` (same size, un-tuned) and `yolo-issia-ball-opt.pt` (153 MB, other dataset) |
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
