"""Stage 2 minimap: the video frame (left) next to a top-down pitch with every player as a dot (right).

Usage: uv run python src/track/minimap.py clip04   (or SNGS-028)
Reads data/tracks/<clip>.json (positions) and the raw tracking file (boxes, for the left side).
Writes outputs/minimap_<clip>.mp4. Same colour = same ID on both sides. The tail behind each dot
is its last second, so wobble (PnLCalib jitter) and jumps (ID switches, bad boxes) are easy to spot.
SoccerNet clips also get the true players from the labels as white rings: dot inside ring = right.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from clips import fps, frame_path, is_soccernet  # same folder
from draw_tracks import colour

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from homography import project  # noqa: E402
from pitch_model import ARC_RADIUS, BOX_DEPTH, LINES, PENALTY_SPOT, POINTS  # noqa: E402
from compare_pnl import GOAL_SIDE, pitch_to_soccernet  # noqa: E402
from eval_soccernet import load_truth  # noqa: E402

PX = 8  # map pixels per meter
X0, X1 = -40, 40  # meters along the goal line: past both touchlines
Y0, Y1 = -8, 110  # meters into the field: behind the goal .. past the far goal line (SoccerNet players reach 94 m)
S = np.array([[PX, 0, -PX * X0], [0, -PX, PX * Y1], [0, 0, 1]])  # meters -> map pixels, as in topdown.py
MAP_W, MAP_H = (X1 - X0) * PX, (Y1 - Y0) * PX  # 640 x 944
HALF_W, HALF_L = 34, 52.5  # ponytail: assumes a 105 x 68 m pitch (SoccerNet's), only to draw the lines
TRAIL_S = 1  # seconds of tail behind each dot


def soccernet_to_pitch(XY, goal):
    """SoccerNet meters (X, Y) -> our pitch meters (x, y): pitch_to_soccernet backwards."""
    if goal == "left":  # X = -52.5 + y, Y = x
        return np.stack([XY[:, 1], XY[:, 0] + 52.5], axis=1)
    return np.stack([-XY[:, 1], 52.5 - XY[:, 0]], axis=1)  # right: X = 52.5 - y, Y = -x


def to_map(xy):
    return project(S, np.atleast_2d(np.asarray(xy, dtype=float))).round().astype(np.int32)


def pitch_background():
    img = np.full((MAP_H, MAP_W, 3), (40, 110, 40), np.uint8)
    white = (255, 255, 255)
    segs = [(POINTS[a][:2], POINTS[b][:2]) for a, b in LINES]
    L = 2 * HALF_L
    segs += [((-HALF_W, 0), (HALF_W, 0)), ((-HALF_W, 0), (-HALF_W, L)), ((HALF_W, 0), (HALF_W, L)),
             ((-HALF_W, HALF_L), (HALF_W, HALF_L)), ((-HALF_W, L), (HALF_W, L))]  # goal lines, touchlines, halfway
    for seg in segs:
        p, q = to_map(seg)
        cv2.line(img, tuple(p), tuple(q), white, 1, cv2.LINE_AA)
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    arc = np.stack([ARC_RADIUS * np.sin(t), PENALTY_SPOT + ARC_RADIUS * np.cos(t)], axis=1)
    cv2.polylines(img, [to_map(arc[arc[:, 1] >= BOX_DEPTH])], False, white, 1, cv2.LINE_AA)
    cv2.circle(img, tuple(to_map((0, HALF_L))[0]), round(9.15 * PX), white, 1, cv2.LINE_AA)  # centre circle
    cv2.circle(img, tuple(to_map((0, PENALTY_SPOT))[0]), 2, white, -1)
    p, q = to_map([POINTS["goalpost_left"][:2], POINTS["goalpost_right"][:2]])
    cv2.line(img, tuple(p), tuple(q), white, 4)  # the goal
    return img


def main(clip):
    tracks = json.load(open(ROOT / "data" / "tracks" / f"{clip}.json"))
    raw_path = ROOT / "data" / "track" / f"{clip}_football-player-detection-v9_botsort.json"
    raw = {f["frame"]: f for f in json.load(open(raw_path))["frames"]}
    bg = pitch_background()
    FPS, trail = fps(clip), round(TRAIL_S * fps(clip))
    truth = {}
    if is_soccernet(clip):
        goal = GOAL_SIDE[clip]
        truth = {f: soccernet_to_pitch(xy, goal) for f, (_, xy, _) in load_truth(clip).items()}
        some = next(iter(truth.values()))  # there and back must land on the same points
        assert np.allclose(soccernet_to_pitch(pitch_to_soccernet(some, goal)[:, :2], goal), some)
    out = ROOT / "outputs" / f"minimap_{clip}.mp4"
    W, H = 960 + MAP_W, max(540, MAP_H)  # 1600 x 944
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    trails = {}  # id -> recent map points
    for fr in tracks["frames"]:
        canvas = np.zeros((H, W, 3), np.uint8)
        # Left: the frame at half size, with the boxes of the players that are on the map.
        img = cv2.resize(cv2.imread(str(frame_path(clip, fr["frame"]))), (960, 540))
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
        for xy in truth.get(fr["frame"], []):
            cv2.circle(m, tuple(to_map(xy)[0]), 7, (255, 255, 255), 1, cv2.LINE_AA)
        trails = {p["id"]: trails.get(p["id"], [])[-trail:] + [to_map((p["x"], p["y"]))[0]] for p in fr["players"]}
        for p in fr["players"]:
            c, pts = colour(p["id"]), np.array(trails[p["id"]])
            cv2.polylines(m, [pts], False, c, 1, cv2.LINE_AA)
            cv2.circle(m, tuple(pts[-1]), 4, c, -1)
            cv2.putText(m, str(p["id"]), tuple(pts[-1] + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        canvas[:MAP_H, 960:] = m
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(tracks['frames'])} frames -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
