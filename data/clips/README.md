# Clips

Source: `~/Downloads/Barcelona 5-1 Feyenoord ｜ Champions League 26⧸27 Match Highlights [ajBizPPlASA].mp4`
(YouTube highlights, 1920×1080, 50000/1001 = 49.95 fps). All clips: same match, same stadium, same
main camera (only its pan/tilt/zoom changes between clips).

Recreate a clip: `src/extract_frames.sh "<source>" <clip> <start> <end>`

| clip | start–end (video) | match clock | frames | fps | resolution | camera motion | camera angle | attacked goal | notes |
|---|---|---|---|---|---|---|---|---|---|
| clip01 | 00:00:31–00:00:43 | 2:42–2:53 | 600 | 49.95 | 1920×1080 | pan + zoom (path ~1980 px, zoom ×1.85) | main cam, high, side-on | right of screen | follows play from halfway line; box fully visible only from ~frame 250 |
| clip02 | 00:01:06–00:01:12 | 8:32–8:38 | 300 | 49.95 | 1920×1080 | pan in first ~1 s, then near-steady (path ~680 px, zoom ×1.07) | main cam, high, side-on | right of screen | |
| clip03 | 00:01:17–00:01:25 | 21:09–21:16 | 400 | 49.95 | 1920×1080 | steady pan (path ~920 px, zoom ×1.06) | main cam, high, side-on | right of screen | box fully visible only from ~frame 230 |
| clip04 | 00:02:09–00:02:15.74 | 23:14–23:21 | 337 | 49.95 | 1920×1080 | slow pan + zoom in (path ~400 px, zoom ×1.42) | main cam, high, side-on | left of screen | trimmed: cut to close-up at old frame 338 |
| clip05 | 00:02:18.20–00:02:24 | 25:11–25:16 | 290 | 49.95 | 1920×1080 | near-static (path ~240 px, zoom ×0.95) | main cam, high, side-on | right of screen | trimmed: started with close-up + crossfade |
| clip06 | 00:02:55–00:03:02 | 36:43–36:50 | 350 | 49.95 | 1920×1080 | slow pan + zoom in (path ~360 px, zoom ×1.28) | main cam, high, side-on | right of screen | |
| clip07 | 00:06:09–00:06:16 | 76:49–76:55 | 350 | 49.95 | 1920×1080 | near-static, slight zoom in at the end (path ~90 px, zoom ×1.09) | main cam, high, side-on | left of screen | free kick: ball starts still on the ground |

Camera motion: static / slow pan / pan + zoom / fast pan. Numbers measured with ORB feature
matching every 5th frame: *path* = total distance (px at 1920×1080) the scene point at the image
centre travels; *zoom* = image scale of last frame vs first.
Camera angle: main cam (high, side-on) / behind goal / other (describe).
Attacked goal: left or right of screen (main cam), near or far (behind goal).
Goalkeeper's right (`+x`): goal on right of screen → far side of the pitch; goal on left → near side.

**Easy clip:** clip07 — least camera motion of all 7 (path ~90 px), whole box + goal visible in
every frame, and the ball starts still on the ground (known kick point for Stage 5). Not truly
static: the ×1.09 zoom means one H for the whole clip would be off by up to ~9% of the distance
from the image centre by the last frame. Calibrate frame 0 first.
Backup: clip05 (near-static, goal on the *other* side → tests the mirroring rule).
