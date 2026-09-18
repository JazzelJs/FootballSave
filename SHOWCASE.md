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

Our own broadcast clip, with no ground truth of any kind. Everything works, and the poses are
**better than Version 1's** — see below.

| | |
|---|---|
| players | 337 frames (6.7 s), 23 tracks, teams assigned |
| posed bodies | **4** — tracks 10 (the shooter), 15, 17, 9 |
| facing | **11.4° / 22.3° / 15.7° / 6.3°** median on the four |
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
| best track | 10.4° / **66.6°** | 6.3° / **23.9°** |
| the shooter | 1.2° (only 4 running samples — noise) | 11.4° / **23.2°** (61 samples) |
| worst track | 29.9° / **104.1°** | 22.3° / **51.8°** |

The medians are comparable. **The tails are two to four times better**, because clip04's tracks are
unbroken (313–337 boxes of 337 frames) while SNGS-043 averaged 5.5 fragments per player — and
SNGS-043's worst frames were always the join hand-overs. Boxes are bigger too: 86–98 px against 78.

One oddity to be honest about: on tracks 10 and 17 the camera-space row has a *lower median* than
pitch space (7.4° vs 11.4°, 8.5° vs 15.7°). That row is handed the offset that minimises its own
error, and clip04's players run in a narrow range of directions, so one fixed offset can fit the
middle well. Its tails give it away — 139.1° at the 95th against pitch space's 23.2°.

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
