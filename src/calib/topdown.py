"""Warp a frame into a top-down map of the penalty area, using H.

Usage: uv run python src/calib/topdown.py annotations/clicks/clip07_00000.json
Saves outputs/topdown_<clip>_<frame>.png
  the warped video frame, seen from above
  magenta = the pitch model drawn straight onto the map (exact by construction)
If H is good, the white painted lines in the warped frame sit under the magenta ones.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from homography import compute_h, load_pairs, project
from pitch_model import ARC_RADIUS, BOX_DEPTH, LINES, PENALTY_SPOT, POINTS

ROOT = Path(__file__).resolve().parents[2]
PX_PER_M = 20
X_RANGE = (-30, 30)   # meters along the goal line shown on the map (goalkeeper's left .. right)
Y_RANGE = (-3, 25)    # meters into the field: a bit behind the goal line .. past the arc
MAP_W = (X_RANGE[1] - X_RANGE[0]) * PX_PER_M   # 1200 px
MAP_H = (Y_RANGE[1] - Y_RANGE[0]) * PX_PER_M   # 560 px


def meters_to_map():
    """3x3 matrix S: pitch meters (x, y) -> map pixels (u, v).

    Layout: +x (goalkeeper's right) points right, the field (+y) points UP the image,
    so the goal line is near the bottom. The map's top-left pixel (0, 0) is the
    pitch point (X_RANGE[0], Y_RANGE[1]).
    """
    # u = PX_PER_M * (x - X_RANGE[0])      (shift, then scale)
    #              v = PX_PER_M * (Y_RANGE[1] - y)      (flip, so bigger y = higher up)
    # Write those two equations as a 3x3 matrix acting on (x, y, 1).
    # The bottom row is [0, 0, 1]: no perspective, so w stays 1.


    newmatrix = np.array([[PX_PER_M, 0, -PX_PER_M * X_RANGE[0]],
                          [0, -PX_PER_M, PX_PER_M * Y_RANGE[1]],
                          [0, 0, 1]])
    return newmatrix

    


def main(clicks_path):
    S = meters_to_map()
    corners = project(S, np.array([[X_RANGE[0], Y_RANGE[1]], [X_RANGE[1], Y_RANGE[0]], [0.0, 0.0]]))
    assert np.allclose(corners, [[0, 0], [MAP_W, MAP_H], [600, 500]]), f"S is wrong: {corners}"

    frame_path = json.load(open(clicks_path))["frame"]
    img = cv2.imread(str(ROOT / frame_path))
    names, pitch_xy, pixels = load_pairs(clicks_path)
    H = compute_h(pitch_xy, pixels)   # meters -> frame pixels

    #  M must take a FRAME pixel to a MAP pixel.
    #   Chain: frame pixel --(H inverted)--> meters --(S)--> map pixel.
    #   Matrices apply right-to-left, like functions: (A @ B) @ p means "B first, then A".
    M = S @ np.linalg.inv(H)

    top = cv2.warpPerspective(img, M, (MAP_W, MAP_H))

    magenta = (255, 0, 255)
    for a, b in LINES:
        p, q = project(S, np.array([POINTS[a][:2], POINTS[b][:2]])).round().astype(int)
        cv2.line(top, tuple(p), tuple(q), magenta, 1, cv2.LINE_AA)
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    arc = np.stack([ARC_RADIUS * np.sin(t), PENALTY_SPOT + ARC_RADIUS * np.cos(t)], axis=1)
    arc = project(S, arc[arc[:, 1] >= BOX_DEPTH]).round().astype(np.int32)
    cv2.polylines(top, [arc], False, magenta, 1, cv2.LINE_AA)

    out = ROOT / "outputs" / f"topdown_{Path(clicks_path).stem}.png"
    cv2.imwrite(str(out), top)
    print(f"Saved {out.relative_to(ROOT)}  ({MAP_W}x{MAP_H} px, {PX_PER_M} px per meter)")


if __name__ == "__main__":
    main(sys.argv[1])
