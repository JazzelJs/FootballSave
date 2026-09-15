"""Stage 2 minimap: the video frame (left) next to a top-down pitch with every player as a dot (right).

Usage: uv run python src/track/minimap.py clip04
Reads data/tracks/<clip>.json (positions) and the raw tracking file (boxes, for the left side).
Writes outputs/minimap_<clip>.mp4. Same colour = same ID on both sides. The tail behind each dot
is its last second, so wobble (PnLCalib jitter) and jumps (ID switches, bad boxes) are easy to spot.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from draw_tracks import FPS, colour  # same folder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from homography import project  # noqa: E402
from pitch_model import ARC_RADIUS, BOX_DEPTH, LINES, PENALTY_SPOT, POINTS  # noqa: E402

PX = 12  # map pixels per meter
X0, X1 = -40, 40  # meters along the goal line: past both touchlines
Y0, Y1 = -8, 44   # meters into the field: behind the goal .. past the furthest clip04 player
S = np.array([[PX, 0, -PX * X0], [0, -PX, PX * Y1], [0, 0, 1]])  # meters -> map pixels, as in topdown.py
MAP_W, MAP_H = (X1 - X0) * PX, (Y1 - Y0) * PX  # 960 x 624
HALF_W = 34  # ponytail: assumes a 68 m wide pitch, only to draw the touchlines; measure it if it matters
TRAIL = 50   # frames of tail behind each dot (1 s)


def to_map(xy):
    return project(S, np.atleast_2d(np.asarray(xy, dtype=float))).round().astype(np.int32)


def pitch_background():
    img = np.full((MAP_H, MAP_W, 3), (40, 110, 40), np.uint8)
    white = (255, 255, 255)
    segs = [(POINTS[a][:2], POINTS[b][:2]) for a, b in LINES]
    segs += [((-HALF_W, 0), (HALF_W, 0)), ((-HALF_W, 0), (-HALF_W, Y1)), ((HALF_W, 0), (HALF_W, Y1))]
    for seg in segs:
        p, q = to_map(seg)
        cv2.line(img, tuple(p), tuple(q), white, 1, cv2.LINE_AA)
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    arc = np.stack([ARC_RADIUS * np.sin(t), PENALTY_SPOT + ARC_RADIUS * np.cos(t)], axis=1)
    cv2.polylines(img, [to_map(arc[arc[:, 1] >= BOX_DEPTH])], False, white, 1, cv2.LINE_AA)
    cv2.circle(img, tuple(to_map((0, PENALTY_SPOT))[0]), 2, white, -1)
    p, q = to_map([POINTS["goalpost_left"][:2], POINTS["goalpost_right"][:2]])
    cv2.line(img, tuple(p), tuple(q), white, 4)  # the goal
    return img


def main(clip):
    tracks = json.load(open(ROOT / "data" / "tracks" / f"{clip}.json"))
    raw_path = ROOT / "data" / "track" / f"{clip}_football-player-detection-v9_botsort.json"
    raw = {f["frame"]: f for f in json.load(open(raw_path))["frames"]}
    bg = pitch_background()
    out = ROOT / "outputs" / f"minimap_{clip}.mp4"
    W, H = 960 + MAP_W, max(540, MAP_H)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    trails = {}  # id -> recent map points
    for fr in tracks["frames"]:
        canvas = np.zeros((H, W, 3), np.uint8)
        # Left: the frame at half size, with the boxes of the players that are on the map.
        img = cv2.resize(cv2.imread(str(ROOT / "data" / "frames" / clip / f"{fr['frame']:05d}.jpg")), (960, 540))
        ids = {p["id"] for p in fr["players"]}
        for b in raw[fr["frame"]]["boxes"]:
            if b["id"] in ids:
                x1, y1, x2, y2 = (np.array(b["xyxy"]) / 2).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), colour(b["id"]), 1)
                cv2.putText(img, str(b["id"]), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour(b["id"]), 1)
        if fr["ball_px"]:
            cv2.circle(img, tuple((np.array(fr["ball_px"]) / 2).astype(int)), 6, (0, 0, 255), 2)
        cv2.putText(img, f"frame {fr['frame']}", (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        canvas[:540, :960] = img

        # Right: the pitch from above, one dot + tail per player.
        m = bg.copy()
        trails = {p["id"]: trails.get(p["id"], [])[-TRAIL:] + [to_map((p["x"], p["y"]))[0]] for p in fr["players"]}
        for p in fr["players"]:
            c, pts = colour(p["id"]), np.array(trails[p["id"]])
            cv2.polylines(m, [pts], False, c, 1, cv2.LINE_AA)
            cv2.circle(m, tuple(pts[-1]), 5, c, -1)
            cv2.putText(m, str(p["id"]), tuple(pts[-1] + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        canvas[:MAP_H, 960:] = m
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(tracks['frames'])} frames -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
