"""Stage 2: the foot pixel of each tracked box -> pitch meters, with inv(H) from Stage 1.

Usage: uv run python src/track/to_pitch.py clip04
Reads data/track/<clip>_football-player-detection-v9_botsort.json and data/camera/<clip>.json.
Writes data/tracks/<clip>.json (tracks.json, format in PLAN.md -> Data formats).
Prints two checks:
  1. your clicked pitch points (annotations/clicks) -> meters, vs where they really are.
     Tests pixels_to_meters alone.
  2. the top speed of each player in m/s. Tests everything together: a sprinting footballer
     tops out around 10 m/s, so anything far above that is a bug, a bad H, or an ID switch.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from homography import load_pairs, project  # noqa: E402  (lives in src/calib)

FPS = 49.95
# Drop person tracks whose average YOLO confidence is below this: on clip04 that removes a pile of
# towels by the post (5 IDs) and a steward behind the goal, all 0.20-0.43, while every real player
# is 0.60 or more. Picked on clip04 alone: check the gap again on other clips.
MIN_CONF = 0.5


def foot_point(xyxy):
    """Box [x1, y1, x2, y2] in pixels -> the foot pixel [u, v].

    Remember: in images, v grows DOWNWARD (row 0 is the top of the picture).
    """
    return np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]])  # middle of the bottom edge


def pixels_to_meters(H, pixels):
    """Pixels -> pitch meters. Only right for points on the grass (z = 0).

    H: (3, 3) array from camera.json. It maps pitch meters -> pixels.
    pixels: (N, 2) array, rows [u, v].
    Returns: (N, 2) array, rows [x, y] in pitch meters.
    """
    return project(np.linalg.inv(H), pixels)  # same maths as meters -> pixels, just the inverse H


def frame_entry(frame, boxes, H):
    """One frame of tracks.json (format: PLAN.md -> Data formats).

    frame: frame number (int).
    boxes: this frame's boxes from the raw tracking file, dicts with "id", "cls", "conf",
           "xyxy". cls is "ball" for the ball, anything else is a person.
    H: (3, 3) array for this frame, or None if PnLCalib found no camera.
    Returns: {"frame": frame,
              "players": [{"id": 7, "team": None, "x": -3.2, "y": 14.8, "visible": True}, ...],
              "ball_px": [u, v] or None}
      - players: every person box, placed with foot_point + pixels_to_meters.
        team = None for now: Stage 2 doesn't know teams yet.
        visible = True: every box is something YOLO saw (later, filled-in gaps get False).
        If H is None we can't place anyone: players = [].
      - ball_px: the centre of the ball box in pixels (the most confident one if there are
        several), or None if there's no ball in this frame.
    """
    players = []
    if H is not None:  # no camera -> we can't place anyone
        for b in boxes:
            if b["cls"] != "ball":
                foot_px = foot_point(b["xyxy"])
                x, y = pixels_to_meters(H, np.array([foot_px]))[0]
                players.append({"id": b["id"], "team": None, "x": x, "y": y, "visible": True})

    # The ball stays in pixels, so it needs no H. Several ball boxes -> keep the most confident.
    balls = [b for b in boxes if b["cls"] == "ball"]
    ball_px = None
    if balls:
        x1, y1, x2, y2 = max(balls, key=lambda b: b["conf"])["xyxy"]
        ball_px = [(x1 + x2) / 2, (y1 + y2) / 2]
    return {"frame": int(frame), "players": players, "ball_px": ball_px}


def main(clip):
    raw = json.load(open(ROOT / "data" / "track" / f"{clip}_football-player-detection-v9_botsort.json"))
    cams = {c["frame"]: np.array(c["H"])
            for c in json.load(open(ROOT / "data" / "camera" / f"{clip}.json"))["frames"]}

    # Check 1: known pitch points. A clicked pixel -> meters should land on the pitch model point.
    errs = []
    for path in sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json")):
        frame = int(path.stem.split("_")[1])
        _, pitch_xy, pixels = load_pairs(path)
        errs += list(np.linalg.norm(pixels_to_meters(cams[frame], pixels) - pitch_xy, axis=1))
    print(f"check 1, clicks -> meters: mean {np.mean(errs):.2f} m, worst {np.max(errs):.2f} m ({len(errs)} points)")

    # A ball track sometimes gets a person label for a few frames: call the whole track "ball".
    ball_ids = {b["id"] for f in raw["frames"] for b in f["boxes"] if b["cls"] == "ball"}
    confs = {}
    for f in raw["frames"]:
        for b in f["boxes"]:
            confs.setdefault(b["id"], []).append(b["conf"])
    unsure = {i for i, c in confs.items() if i not in ball_ids and np.mean(c) < MIN_CONF}
    print(f"dropped {len(unsure)} person tracks with mean confidence < {MIN_CONF}: {sorted(unsure)}")
    frames = []
    for f in raw["frames"]:
        boxes = [dict(b, cls="ball") if b["id"] in ball_ids else b for b in f["boxes"] if b["id"] not in unsure]
        frames.append(frame_entry(f["frame"], boxes, cams.get(f["frame"])))
    out = ROOT / "data" / "tracks" / f"{clip}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"clip": clip, "fps": FPS, "frames": frames}, open(out, "w"))
    print(f"{len(frames)} frames -> {out}")

    tracks = {}  # id -> list of (frame, x, y), read back from what goes into tracks.json
    for fr in frames:
        for p in fr["players"]:
            tracks.setdefault(p["id"], []).append((fr["frame"], p["x"], p["y"]))

    # Check 2: top speed per player. Measured every 10th position (0.2 s): frame-to-frame, a 10 cm
    # wobble of the box would already read as 5 m/s.
    print("check 2, top speed per ID:")
    for tid, t in sorted(tracks.items(), key=lambda kv: -len(kv[1])):
        n = len(t)
        t = np.array(t)[::10]
        if len(t) < 2:
            continue
        speed = np.linalg.norm(np.diff(t[:, 1:], axis=0), axis=1) / (np.diff(t[:, 0]) / FPS)
        i = speed.argmax()
        print(f"  #{tid:>3}: {n:>3} frames, starts at ({t[0, 1]:6.1f}, {t[0, 2]:5.1f}) m, "
              f"top {speed[i]:5.1f} m/s around frame {int(t[i + 1, 0])}")


if __name__ == "__main__":
    main(sys.argv[1])
