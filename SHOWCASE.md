# Two versions to show

Both run the same code. The difference is what the clip has behind it. Start the server once from
the repo root and open either URL:

```
python3 -m http.server 8000
```

Buttons in both: `pose view` · `orbit view` · `goal view` (`c`) · `show gaps` (`g`).

---

## Version 1 — SNGS-043: the complete pipeline

`http://localhost:8000/src/viewer/?clip=SNGS-043` — **scrub to 24.3 s, press `c`.**

Every stage is live at once. This is the one to lead with.

| | |
|---|---|
| players | 750 frames, positions **0.48 m** median error vs ground truth |
| teams | **89%** of outfield players given the right team |
| posed bodies | **4** — tracks 1131 (the shooter), 284, 17, 171 |
| facing | **10.4° / 15.2° / 29.9°** median on the three running players |
| pose accuracy | **113 mm** PA-MPJPE (measured on KTH, the close-up best case) |
| ball in 3D | **117 km/h**, 3.70 px reprojection, **inside the post** ✅ |

**The 16 frames that matter: 602–617.** That is the only window where the posed bodies and the
flying ball overlap (poses run 375–625, the shooter 560–620, the ball 602–617 = 0.64 s). Four
bodies, the rest as team-coloured capsules, and a ball with real height.

**What to say about it:** the goalkeeper's view is reconstructed from *one* broadcast camera. The
ball's height was never measured — it was inferred by asking which thrown ball would have been
seen at those pixels, and the answer independently puts the shot inside the posts, which is the
only fact in the whole stage that nobody annotated.

**Honest limits, if asked.** The ball is yellow (true 3D height) for 0.64 s only; before the kick
it is white and sitting on the grass, which is correct, and after the flight there is nothing to
show. 16–17% of players are missed, and after the goal at frame 634 that jumps to 33% because the
detector only knows players who are *playing* — the celebration empties the pitch.

---

## Version 2 — clip04: our own clip, end to end

`http://localhost:8000/src/viewer/?clip=clip04` — **scrub to ~4.2 s.**

Our own broadcast clip, with no ground truth of any kind. The ball still works.

| | |
|---|---|
| players | 337 frames (6.7 s), 23 tracks, teams assigned |
| posed bodies | **none yet** — the viewer says "capsules only, no poses" |
| ball in 3D | **106 km/h**, 9.20 px reprojection, **0.87 m outside the post** ✅ |

**The point of this one is the near miss.** clip04 is a shot that misses, and the fit — with no
labels, no ground truth, and a kick frame found inside a detection gap — puts the ball 4.53 m from
the centre of the goal, against a post at 3.66 m. It got the outcome right on a clip that could
not tell it the answer.

**Frames 196–227** are the flight (0.64 s). The kick is at 196; contact is actually at 194–195,
inside the ball detector's blind spot, because a ball being struck is motion-blurred.

### To finish Version 2 — one Kaggle session

Four tracks are ready and they are a *better* pose target than SNGS-043: boxes **86–98 px** tall
against SNGS-043's 78 px, and the tracks are **unbroken** (313–337 boxes of 337 frames, no joined
fragments, so no hand-over errors).

```
uv run python src/pose/export_kaggle_inputs.py clip04 10 15 17 9
```

Then follow the five steps it prints. Track **10 is the shooter**. After the run:

```
uv run python src/pose/export_smpl_mesh.py --clip clip04 --track 10 --smooth-yaw 5
```

and add `{ clip: 'clip04', trackId: 10, file: 'pose/mesh/pose_10.smpl' }` to `ALL_POSES` in
`src/viewer/index.html`. `orient.py` needs the per-frame camera and `GOAL_SIDE['clip04']`, both of
which already exist, so facing works with no new code.

**Two things to check rather than assume.** clip04 has no ground truth, so PA-MPJPE is not
available — the only facing check is the direction-of-travel proxy, whose own noise floor is about
10°. And clip04 runs at **49.95 fps**, so `--smooth-yaw 5` is 0.1 s here versus 0.2 s on SNGS-043;
re-measure that window instead of inheriting it.

---

## Supporting material already rendered

| file | what |
|---|---|
| `outputs/ball_fit_SNGS-043.mp4` | the fitted flight on the broadcast frames |
| `outputs/ball_fit_clip04.mp4` | same for our clip |
| `outputs/ball_fit_clip04_000210.jpg` | one frame: fit on the ball, trajectory past the post |
| `outputs/ball_shadow_SNGS-043.png` | C3 — and why it cannot measure height on this clip |
| `outputs/facing_SNGS-043_000602.jpg` | facing arrows at the kick |
| `outputs/minimap_<clip>.mp4` | the 2D view, ground truth as white rings |
