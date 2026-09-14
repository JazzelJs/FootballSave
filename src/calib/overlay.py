"""Draw the pitch model onto a frame through H, to see where H is right and wrong.

Usage: uv run python src/calib/overlay.py annotations/clicks/clip07_00000.json
Saves outputs/overlay_<clip>_<frame>.png (full resolution, zoom in to inspect).
  magenta lines   = pitch lines where H puts them (should sit ON the white lines)
  red cross       = where you clicked each point
  green circle    = where H puts that point
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from homography import compute_h, load_pairs, project
from pitch_model import ARC_RADIUS, BOX_DEPTH, LINES, PENALTY_SPOT, POINTS

ROOT = Path(__file__).resolve().parents[2]


def to_px(H, xy):
    return [tuple(int(round(c)) for c in p) for p in project(H, np.asarray(xy, dtype=np.float64))]


def main(clicks_path):
    if not LINES:
        sys.exit("LINES in pitch_model.py is still empty: fill it in first.")
    frame_path = json.load(open(clicks_path))["frame"]
    img = cv2.imread(str(ROOT / frame_path))
    names, pitch_xy, pixels = load_pairs(clicks_path)
    H = compute_h(pitch_xy, pixels)  # all points: this is the H you'd actually use

    magenta = (255, 0, 255)
    for a, b in LINES:
        p, q = to_px(H, [POINTS[a][:2], POINTS[b][:2]])
        cv2.line(img, p, q, magenta, 1, cv2.LINE_AA)

    t = np.linspace(-np.pi / 2, np.pi / 2, 200)  # arc around the spot, keep the part beyond the box
    arc = np.stack([ARC_RADIUS * np.sin(t), PENALTY_SPOT + ARC_RADIUS * np.cos(t)], axis=1)
    arc = arc[arc[:, 1] >= BOX_DEPTH]
    cv2.polylines(img, [np.array(to_px(H, arc), dtype=np.int32)], False, magenta, 1, cv2.LINE_AA)

    for (u, v), (pu, pv) in zip(pixels.round().astype(int), to_px(H, pitch_xy)):
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.circle(img, (pu, pv), 5, (0, 255, 0), 1, cv2.LINE_AA)

    out = ROOT / "outputs" / f"overlay_{Path(clicks_path).stem}.png"
    cv2.imwrite(str(out), img)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
