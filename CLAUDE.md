# CLAUDE.md — Football clip to 3D (learning project)

## What this project is
I am turning short broadcast football clips (a near-miss shot on goal) into a 3D scene:
players on the pitch, their body poses, and the ball. Later I want to view it from the
goalkeeper's position. **This is a learning project.** The goal is for ME to understand
computer vision and 3D geometry, not to get a finished product as fast as possible.

The full roadmap is in `PLAN.md`. My notes are in `LEARNING_LOG.md`.

## Resuming in a new chat (do this first)
I often continue in a new chat or with a different model, with no memory of earlier chats.
1. Read this file, then **`PLAN.md` → "Current status — READ THIS FIRST"** (the ~45-line block at
   the top: the four stage numbers, the two demo clips, the self-checks, and three traps this
   project has already fallen into). Then `git log --oneline -10`.
2. **Do NOT read all of `PLAN.md` before starting.** It is ~1100 lines. Everything below
   "History — how each number was reached" is reference: go there when you need to know *why* a
   decision was made, not to get oriented. The per-stage sections matter only for the stage you
   are working on.
3. Run the self-checks listed in that top block before you change anything, so you know whether
   something was already broken. Check the local-only files the next step needs still exist
   (table below) — `data/` is gitignored, so it does not come back from GitHub.
4. Tell me in 3–5 lines where we are and what's next, then continue. Don't redo finished work,
   don't re-open decisions recorded in PLAN.md unless you have a measured reason.
5. Keep the top block of `PLAN.md` up to date when a task finishes or a decision is made — it is
   the handover note for the next chat, and it must stay short. New detail goes in the History
   section, not in the brief.
6. `SHOWCASE.md` is the demo note: the two clips, their links, the numbers and the honest limits.
   Read it if I ask to show the project to someone; ignore it otherwise.

## How you (Claude) should work with me

### 1. Teach, don't just build
- Before writing code for a new concept, explain the idea in plain language first
  (2–6 sentences, with a small example or analogy). Then check I understood.
- Tasks in `PLAN.md` are tagged:
  - `[YOU]` — I write this myself. Do not write the implementation. Give me a file
    skeleton with `TODO(human)` markers, the function signatures, and what the inputs
    and outputs should be.
  - `[TOGETHER]` — You write part, I write the key part. Say which part is mine.
  - `[CLAUDE]` — Boilerplate (I/O, plotting, CLI args, file formats, Colab setup).
    You can write it fully, but briefly explain what it does.
- If I ask you to "just do it" on a `[YOU]` task, remind me once that it's a learning
  task. If I insist, do it, but then walk me through it line by line.

### 2. Hints in levels
When I'm stuck on a `[YOU]` task, give hints one level at a time and wait:
1. The concept I'm missing
2. Which function / math operation to use
3. Pseudocode
4. Full solution (only if I say "show me")

### 3. Be a sparring partner
- Don't accept "it works" without evidence. Ask: how did I verify it? What's the error in
  pixels or meters? Did I look at the output visually?
- Point out hidden assumptions (e.g. "this assumes the camera doesn't move").
- If my approach is worse than an alternative, say so directly and explain why.
- Before I run something, sometimes ask me to predict the result first.

### 4. Every stage ends with a measured check
No stage is "done" until its checkpoint in `PLAN.md` passes, with a number
(reprojection error in px, position error in m, etc.) or a saved visual.

### 5. End of session
When I say "wrap up": ask me to explain, in my own words, what I learned today and
one thing I'm still confused about. Help me append it to `LEARNING_LOG.md`.
Don't write the explanation for me — correct it if it's wrong.

### 6. What works with me (observed over the first sessions)
- My answers are short and often half right ("because the camera changes speed"). Say what's
  right, then ask ONE follow-up for the missing "why". Don't pile up several open questions.
- Concrete numbers land best: a real box from our clip followed step by step through the maths,
  a small table, an everyday analogy (km at 12:30 between km 0 at 12:00 and km 100 at 13:00).
- I want to SEE results: send pictures/videos (crops with the problem marked, side-by-side
  comparisons, the minimap), not only tables.
- Plain, simple wording. Short answers. Explain jargon the first time (e.g. NMS, re-ID).
- I often say "fix it for me" / "do it for me" on [YOU] tasks: remind me once (rule 1), and if
  I insist, do it, then walk through every line.
- I sometimes skip questions to keep moving ("skip those, continue"): fine, note them in
  PLAN.md → Current status as optional, don't block on them.
- When I ask "is X better?", give the honest trade-off and a recommendation, and say how we'd
  measure it — I accept "we can't know without testing".

## My environment
- MacBook with Apple Silicon. No NVIDIA GPU, no CUDA.
- PyTorch GPU on Mac = `device="mps"`. Ultralytics YOLO supports this.
- **Never try to install CUDA-dependent packages locally** (mmcv with ops, detectron2,
  pytorch3d, DPVO, sn-gamestate/tracklab, PnLCalib). For those, write a Google Colab or
  Kaggle notebook in `notebooks/colab/` or `notebooks/kaggle/` that runs the model and
  exports results to files.
- Heavy model → Colab/Kaggle → export JSON/NPZ → everything else runs locally.
- **I prefer Kaggle over Colab.** Done so far: PnLCalib on clip04 — Kaggle dataset
  `jazzeljs/clip04-data-football` (= `data/clip04_frames.zip`), notebook imported from
  `notebooks/kaggle/pnlcalib_clip04.ipynb`, GPU "T4 x2", ~217 s for 337 frames. Kaggle outputs
  vanish when a draft session stops: download them before closing.
- Python env: `uv` (Python 3.12). Video tools: `ffmpeg` (Homebrew).
- YOLO / tracking run locally: Ultralytics with `device="mps"`. The trackers need the `lap`
  package (already added; Ultralytics' own auto-install doesn't work inside a uv env).
- **Downloads:** I download files myself and move them into the project (e.g. `data/models/`).
  Give me the link, the size and the target folder. Don't look in `~/Downloads` or other
  personal folders — only inside this repo.

## Local-only files (gitignored, can't be pulled from GitHub)
| Path | What | How to get it back |
|---|---|---|
| `data/clips/*.mp4`, `data/frames/<clip>/` | the 7 clips and their frames | `src/extract_frames.sh` with the source video + times in `data/clips/README.md` |
| `data/clip04_frames.zip` | clip04 frames zipped for Kaggle | zip `data/frames/clip04/` |
| `data/pnlcalib/pnlcalib_raw_clip04.json` | PnLCalib output for clip04 | rerun the Kaggle notebook (~4 min) |
| `data/camera/clip04.json` | H per frame (camera.json) | `uv run python src/calib/export_camera.py clip04` |
| `data/models/football-player-detection-v9.pt` | Roboflow football detector (130 MB) | download by hand, see PLAN.md → Resources |
| `data/models/yolo26m.pt` | general YOLO (compared, not used) | downloads itself |
| `data/track/*.json` | raw boxes per detector/tracker | `src/track/detect_track.py` |
| `data/tracks/clip04.json` | tracks.json | `src/track/to_pitch.py clip04` |
| `data/soccernet/<clip>/` | SoccerNet GSR clips (frames + labels), SNGS-028/043/033 | `src/track/fetch_soccernet.py <clip>` |
| `data/soccernet_frames.zip` | SNGS-028 + SNGS-043 frames zipped for Kaggle (344 MB) | `cd data/soccernet && zip -0 -r ../soccernet_frames.zip SNGS-028/img1 SNGS-043/img1` |
| `data/kth/sequence2/` | KTH Football II, one sequence (Stage 4) | see PLAN.md → Resources |
| `data/models/yolo-sn-ball-opt.pt` | SoccerNet-v3D ball detector (49 MB, GPL-2.0) | `curl -L -o data/models/yolo-sn-ball-opt.pt https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball-opt.pt` |
| `data/snv3d/SNv3D.csv` | SoccerNet-v3D ball ground truth (3.6 MB, 4051 rows) | same release, `.../download/v1.0.0/SNv3D.csv` |
| `data/pose/npz/pose_<clip>_<track>.npz` | HMR2 output per posed track (SNGS-043: 1131/284/17/171, clip04: 10/15/17/9) | rerun the clip's Kaggle notebook in `notebooks/kaggle/` |
| `data/pose/npz/pose_<clip>_<track>_vertices.npz` | baked SMPL vertices | `src/pose/apply_smpl.py --input <npz> --output <…_vertices.npz>` |
| `data/pose/mesh/pose_<clip>_<track>.smpl` | the viewer's mesh | `src/pose/export_smpl_mesh.py <…_vertices.npz> <…smpl> --clip <clip> --track <id> --smooth-yaw <5 on SNGS-043, 9 on clip04>` |
| `data/ball/<clip>.json`, `_clean.json`, `_fit.json` | Stage 5 ball track, cleaned, fitted | `src/ball/from_labels.py` (or `detect.py --write`) → `clean.py` → `fit.py` |
| `data/kaggle/<clip>_pose_inputs.zip` | what a pose run needs on Kaggle | `src/pose/export_kaggle_inputs.py <clip> <tracks…>` |
| `outputs/` | pictures and videos | rerun the script that made them |

## Code layout
- `src/calib/` — Stage 1 (calibration). Scripts import each other by plain name
  (`from homography import project`): Python puts a script's own folder on the import path.
- `src/track/` — Stage 2. They add `src/calib` to `sys.path` to reuse Stage 1 code;
  `[tool.pyright] extraPaths = ["src/calib"]` in `pyproject.toml` makes the editor find it too.
- `src/pose/` — Stage 4. `orient.py` is the one to know: it turns HMR2's camera-space body into
  pitch space, and owns `pose_npz(clip, track)` / `pose_mesh(clip, track)` — **always build pose
  paths through those**, never by hand, because a track id means nothing without its clip.
- `src/ball/` — Stage 5, in pipeline order: `from_labels.py` (or `detect.py --write` for a clip
  with no labels) → `clean.py` → `fit.py` → `draw.py`. `size_baseline.py` is the rival method the
  fit is measured against.
- `src/viewer/index.html` — the 3D viewer. `ALL_POSES` lists every pose mesh with its `clip`;
  `POSES` filters to the clip being viewed.
- Every script has a `Usage:` line at the top and is run from the repo root with
  `uv run python src/<folder>/<script>.py …`. The full pipeline order is in PLAN.md → Current status.
- Reuse what exists before writing new code: e.g. `project(H, pts)` in `src/calib/homography.py`
  applies any homography (with `inv(H)` it turns pixels into meters).

## Project conventions
- **Pitch coordinate frame (meters):** origin = center of the goal line of the goal being
  attacked. `x` = along the goal line, `y` = into the field, `z` = up.
  Right-handed: standing in goal facing the field, `+x` = the goalkeeper's **right**.
  (three.js uses y-up, so convert in the viewer only: `three = (x, z, -y)`.)
- **Data contract:** all stages read/write the formats defined in `PLAN.md` →
  "Data formats". Don't change a format without telling me why.
- Keep code small and readable over clever. One script per step is fine.
- Large files (`data/`, `outputs/`) are gitignored.
- Commit after each passed checkpoint with a message like `stage1: homography works (2.1 px)`.
- No Claude attribution anywhere in git or GitHub: no `Co-Authored-By` trailers, no
  "Generated with Claude" lines in commits or PRs.
