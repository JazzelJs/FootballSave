# Three versions to show

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

Our own broadcast clip, with no ground truth of any kind. Everything works, and the poses are
**better than Version 1's** — see below.

| | |
|---|---|
| players | 337 frames (6.7 s), 23 tracks, teams assigned |
| posed bodies | **4** — tracks 10 (the shooter), 15, 17, 9 |
| facing | **12.0° / 21.0° / 15.7° / 6.4°** median on the four (`--smooth-yaw 9`) |
| ball in 3D | **106 km/h**, 9.20 px reprojection, **0.87 m outside the post** ✅ |

**The point of this one is the near miss.** clip04 is a shot that misses, and the fit — with no
labels, no ground truth, and a kick frame found inside a detection gap — puts the ball 4.53 m from
the centre of the goal, against a post at 3.66 m. It got the outcome right on a clip that could
not tell it the answer.

**Frames 196–227** are the flight (0.64 s). The kick is at 196; contact is actually at 194–195,
inside the ball detector's blind spot, because a ball being struck is motion-blurred.

### Why Version 2's poses came out better

Same model, same code, better input — and the improvement lands exactly where it was predicted to.

| facing error, median / 95th | SNGS-043 | clip04 |
|---|---|---|
| best track | 10.4° / **66.6°** | 6.4° / **21.6°** |
| the shooter | 1.2° (only 4 running samples — noise) | 12.0° / **25.0°** (61 samples) |
| worst track | 29.9° / **104.1°** | 21.0° / **51.4°** |

The medians are comparable. **The tails are two to four times better**, because clip04's tracks are
unbroken (313–337 boxes of 337 frames) while SNGS-043 averaged 5.5 fragments per player — and
SNGS-043's worst frames were always the join hand-overs. Boxes are bigger too: 86–98 px against 78.

One oddity to be honest about: on tracks 10 and 17 the camera-space row has a *lower median* than
pitch space (7.4° vs 12.0°, 8.5° vs 15.7°). That row is handed the offset that minimises its own
error, and clip04's players run in a narrow range of directions, so one fixed offset can fit the
middle well. Its tails give it away — 139.1° at the 95th against pitch space's 25.0°.

**On the smoothing window**, if anyone asks why it is 9 here and 5 on SNGS-043: it is the same
0.2 s, and clip04 runs at twice the frame rate. Sweeping it does not help — the facing error falls
monotonically all the way to a 0.8 s window, because the travel-direction reference is itself
smooth and rewards any smoothing at all. Across w=1..15 the median moves under 3° on every track,
so the window is chosen on physics, not fitted.

---

## Version 3 — SNGS-028: a miss, with ground truth

`http://localhost:8000/src/viewer/?clip=SNGS-028` — **scrub to ~15.5 s.**

The cell the other two do not cover. clip04 misses but has no ground truth; SNGS-043 has ground
truth but scores. This one **misses and is labelled**, so the "outside the posts" claim can be
checked rather than believed.

| | |
|---|---|
| players | 750 frames, positions **0.70 m** median error, **96%** team correct (the best of the three) |
| posed bodies | **none yet** — one Kaggle session, `notebooks/kaggle/football3d_sngs028_poses.ipynb` |
| ball in 3D | **78 km/h**, 8.57 px, crossing at **x −4.16 m, z 2.20 m** — wide of the post and under the bar ✅ |

`action_class` is "Shots off target", and the fit independently puts the ball **0.5 m outside the
near post, 0.24 m under the bar**. Three clips, three outcomes, three times the fit agreed with
what actually happened without being told.

**Say this if you show it, because it is the weakest of the three fits:** only **9 usable frames**
against 6 unknowns, and our camera disagrees with the labels by **5.04 m** at the kick point
(1.55 m on SNGS-043). Treat 78 km/h as indicative, not measured.

### What this clip taught us, which is the better story

The first attempt fitted at **24.79 px** and I blamed the ball for being airborne. Wrong on both
counts:
- The test used — pixel motion versus the motion a grass-bound ball would make — is **confounded
  by camera pan**. A broadcast camera follows the ball, so a moving ball sits almost still in the
  image. Low pixel motion with high grass motion is *tracking*, not *flying*.
- The real fault was our own window rule. It ends the flight where the pixel track **reverses** —
  which is the ball hitting the net. **A shot off target never reverses**, so it ran to the end of
  the data and the fit degraded 8.57 → 24.79 px as the window grew.
- And the late frames were bad annotation, not bad physics: from frame 396 the steps alternate
  64, 7, 37, 74, 2, 71, and the box stays a constant 20–21 px while the ball flies 40 m away,
  where SNGS-043's shrank 24 → 11 px.

### To finish Version 3 — one Kaggle session

**Crops are the biggest of all three clips: 104–150 px median** (clip04 86–98, SNGS-043 78),
because this camera is tighter, and Stage 4's ceiling was crop size.

```
uv run python src/pose/export_kaggle_inputs.py SNGS-028 14 899 904 493 --window 275 425
```

**Seconds 11–17 (frames 275–425), not the whole clip:** SNGS-028's camera fails on 69 frames and
the two long runs are 97–123 and 236–264, which that window clears while still containing the shot
at 387. Track **14 is the shooter**. About 520 crops, less than half the clip04 run.

The catch: its tracking is the weakest of the three — **6.1 fragments per real player** against
SNGS-043's 5.5, and 18% of true players missed. Expect more ID trouble, not less.

---

## Supporting material already rendered

| file | what |
|---|---|
| `outputs/ball_fit_SNGS-043.mp4` | the fitted flight on the broadcast frames |
| `outputs/ball_fit_clip04.mp4` | same for our clip |
| `outputs/ball_fit_SNGS-028.mp4` | same for the shot off target |
| `outputs/ball_fit_clip04_000210.jpg` | one frame: fit on the ball, trajectory past the post |
| `outputs/ball_shadow_SNGS-043.png` | C3 — and why it cannot measure height on this clip |
| `outputs/facing_SNGS-043_000602.jpg` | facing arrows at the kick |
| `outputs/minimap_<clip>.mp4` | the 2D view, ground truth as white rings |
