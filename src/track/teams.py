"""Stage 2: which team each track belongs to, from the shirt colour in the box.

Usage: uv run python src/track/teams.py SNGS-043   (or clip04)
Reads data/tracks/<clip>.json (positions + box_px) and the clip's frames, writes the same file back
with "team" filled in for every player: "A", "B" or "other" (referees, goalkeepers, anything odd).

Why not use the detector's classes: on our match it calls the referee a "player" 97% of the time
(PLAN.md -> Known issues). The shirt colour is the real signal.

Measure it with: uv run python src/track/eval_soccernet.py SNGS-043   -> (D) teams.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from clips import frame_path  # same folder

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = 20  # frames to look at per track: the median over 20 crops beats one lucky crop
HUE_BINS = 12       # hue histogram: 15 degrees per bin, enough to tell two kits apart
OTHER_FACTOR = 2.0  # a track further than this times the median distance from its cluster = "other"


def torso_patch(img, box_px):
    """The middle of the upper half of a player box: mostly shirt, little grass, no head.

    img: the frame (BGR, as cv2.imread gives it).
    box_px: [x1, y1, x2, y2] from tracks.json.
    Returns: a small BGR image (h, w, 3), or None if the box is too small to crop.
    """
    x1, y1, x2, y2 = (round(v) for v in box_px)
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None  # a far-away player, a few pixels tall: no usable colour
    top, bottom = y1 + round(0.25 * h), y1 + round(0.55 * h)  # shoulders .. waist
    left, right = x1 + round(0.25 * w), x2 - round(0.25 * w)  # middle half: skips the arms and the background
    patch = img[max(top, 0):bottom, max(left, 0):right]
    return patch if patch.size else None


def shirt_colour(patch):
    """One colour for this torso patch, as something two teams can be told apart by.

    patch: small BGR image from torso_patch.
    Returns: a small 1-D numpy array (the same length for every patch), or None if the patch is
             useless (e.g. almost all grass).
    Think about:
      - BGR averages badly: dark red and bright red are far apart as numbers, same shirt to us.
        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV) gives hue (colour), saturation (how strong),
        value (how bright). Hue survives shadow and floodlights much better.
      - Hue is a circle: red is both 0 and 179 in OpenCV. An average over the circle can land
        on the opposite side. A histogram avoids that problem entirely.
      - The patch still holds some grass, some skin, maybe an arm. A median or a histogram peak
        is less easily dragged off than a mean.
      - White and black shirts have no meaningful hue at all: saturation/value must carry those.
    (Claude wrote this on my request, walked through line by line.)
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()

    grass = (hue >= 30) & (hue <= 85) & (sat > 40)  # the pitch: green and colourful enough to be sure
    keep = ~grass  # NOT "and bright enough": a black referee kit is dark and still a shirt
    if keep.sum() < 0.15 * len(hue):
        return None  # almost all grass: the box is mostly pitch, no shirt to look at

    hue, sat, val = hue[keep], sat[keep], val[keep]
    # A histogram instead of an average, because hue is a circle: red is both 0 and 179, and the
    # average of those two is cyan. Each pixel votes with its saturation, so washed-out pixels
    # (skin, white socks, shadow) barely count and a strong shirt colour dominates.
    # .astype(int) matters: hue is uint8, so hue * 12 wraps around at 255 and blue lands in red's bin.
    hist = np.bincount(hue.astype(int) * HUE_BINS // 180, weights=sat / 255.0, minlength=HUE_BINS)[:HUE_BINS]
    # Let each bin bleed into its neighbours, wrapping around the circle (np.roll), so that two
    # slightly different reds landing in bins 11 and 0 still look alike instead of opposite.
    hist = 0.25 * np.roll(hist, 1) + 0.5 * hist + 0.25 * np.roll(hist, -1)
    hist = hist / max(hist.sum(), 1e-6)
    # Plus how colourful and how bright the shirt is: that's all a white or black kit has.
    return np.concatenate([hist, [np.median(sat) / 255.0, np.median(val) / 255.0]])


def assign_teams(colours):
    """Split the tracks into two teams by their colour, everything else into "other".

    colours: {track id: the array shirt_colour gave for that track (its median over SAMPLES frames)}
    Returns: {track id: "A" / "B" / "other"}
    Think about:
      - Two teams = two clusters. cv2.kmeans(data, K=2, ...) does it (scipy/sklearn aren't installed).
      - Which cluster becomes "A" is arbitrary and may differ per clip. That's fine.
      - Referees and the two goalkeepers wear their own colours: a track far from BOTH cluster
        centres should be "other" rather than forced into a team.
      - There are ~11 tracks per team and 1-3 referees, so the clusters are lopsided, not even.
    (Claude wrote this on my request, walked through line by line.)
    """
    ids = list(colours)
    data = np.stack([colours[i] for i in ids]).astype(np.float32)
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    # attempts=10: k-means starts from random centres, so run it 10 times and keep the tightest result.
    _, labels, centres = cv2.kmeans(data, 2, None, stop, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    # How far each track sits from the centre it was given. A referee or a goalkeeper gets put in one
    # of the two teams anyway (k-means has nowhere else to put them), but sits much further out.
    dist = np.linalg.norm(data - centres[labels], axis=1)
    limit = OTHER_FACTOR * np.median(dist)
    return {i: ("other" if d > limit else "AB"[l]) for i, l, d in zip(ids, labels, dist)}


def check_teams():
    """Fake players with known shirts: two teams of solid colour on grass, plus a referee in black."""
    rng = np.random.default_rng(0)

    def fake(bgr):  # a 40 x 20 player box: shirt on top, grass around it
        img = np.zeros((40, 20, 3), np.uint8)
        img[:, :] = (40, 110, 40)  # grass
        img[8:24, 3:17] = bgr  # the shirt, where torso_patch looks
        return np.clip(img + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)

    shirts = {1: (30, 30, 200), 2: (35, 25, 210), 3: (40, 35, 195),  # red team
              4: (200, 60, 30), 5: (210, 55, 40), 6: (190, 70, 35),  # blue team
              7: (20, 20, 20)}  # referee in black
    colours = {}
    for tid, bgr in shirts.items():
        patch = torso_patch(fake(bgr), [0, 0, 20, 40])
        assert patch is not None and patch.size, "torso_patch should find the shirt in a 20x40 box"
        c = shirt_colour(patch)
        assert c is not None, "a solid shirt must give a colour"
        colours[tid] = c
    teams = assign_teams(colours)
    assert teams[1] == teams[2] == teams[3], "the three red shirts belong together"
    assert teams[4] == teams[5] == teams[6], "the three blue shirts belong together"
    assert teams[1] != teams[4], "red and blue are different teams"
    assert {teams[1], teams[4]} == {"A", "B"}, "the two teams are called A and B"


def main(clip):
    path = ROOT / "data" / "tracks" / f"{clip}.json"
    tracks = json.load(open(path))

    seen = {}  # track id -> [(frame, box_px), ...]
    for fr in tracks["frames"]:
        for p in fr["players"]:
            seen.setdefault(p["id"], []).append((fr["frame"], p["box_px"]))

    # Sample SAMPLES frames per track, spread over its life, and remember which frames to read.
    wanted = {}  # frame -> [(track id, box_px), ...]
    for tid, rows in seen.items():
        for i in np.linspace(0, len(rows) - 1, min(SAMPLES, len(rows))).round().astype(int):
            frame, box = rows[i]
            wanted.setdefault(frame, []).append((tid, box))

    per_track = {}  # track id -> list of colours
    for frame in sorted(wanted):  # read each frame once: they are big JPEGs
        img = cv2.imread(str(frame_path(clip, frame)))
        for tid, box in wanted[frame]:
            patch = torso_patch(img, box)
            if patch is None:
                continue
            c = shirt_colour(patch)
            if c is not None:
                per_track.setdefault(tid, []).append(np.asarray(c, float))

    colours = {tid: np.median(np.stack(cs), axis=0) for tid, cs in per_track.items() if cs}
    if not colours:
        print("shirt_colour isn't written yet (TODO(human)): leaving team = null")
        return
    check_teams()
    teams = assign_teams(colours)
    if teams is None:
        print("assign_teams isn't written yet (TODO(human)): leaving team = null")
        return

    for fr in tracks["frames"]:
        for p in fr["players"]:
            p["team"] = teams.get(p["id"], "other")
    json.dump(tracks, open(path, "w"))

    counts = {}
    for t in teams.values():
        counts[t] = counts.get(t, 0) + 1
    print(f"{len(seen)} tracks, colour found for {len(colours)} -> {counts} (tracks per team) -> {path}")


if __name__ == "__main__":
    main(sys.argv[1])
