"""Compute H from clicked points and measure how well it fits.

Convention: H maps pitch meters (x, y) -> image pixels (u, v).
(The other direction, pixels -> meters, is np.linalg.inv(H).)

Usage: uv run python src/calib/homography.py annotations/clicks/clip07_00000.json
"""
import json
import sys

import cv2
import numpy as np

from pitch_model import POINTS

HELD_OUT = ["arc_left", "goal_area_front_right"]  # not used to compute H, only to test it


def load_pairs(clicks_path):
    """Names that are in both files -> (names, pitch_xy (N,2) meters, pixels (N,2))."""
    clicks = json.load(open(clicks_path))["points_px"]
    names = [n for n in POINTS if n in clicks]
    pitch_xy = np.array([POINTS[n][:2] for n in names], dtype=np.float64)
    pixels = np.array([clicks[n] for n in names], dtype=np.float64)
    return names, pitch_xy, pixels


def compute_h(pitch_xy, pixels):
    """Return the 3x3 H that maps pitch_xy (meters) onto pixels, fitted to all given points."""
    # Default method = least squares over all points. Not cv2.RANSAC: it would silently
    # drop points it dislikes, and we want to SEE bad clicks.
    homography = cv2.findHomography(pitch_xy, pixels)[0]

    return homography


def project(H, pitch_xy):
    """Map (N,2) pitch meters -> (N,2) pixels with H, by hand (idea 3)."""
    ones = np.ones((len(pitch_xy), 1))
    pts = np.hstack([pitch_xy, ones])   # N x 3, rows (x, y, 1)
    out = pts @ H.T                     # N x 3, rows (a, b, w); H.T because points are rows
    return out[:, :2] / out[:, 2:3]     # divide by w, kept as an N x 1 column
#out[:, :2] means "all rows, columns 0 and 1": the a and b values, [[2, 30]].
#out[:, 2:3] means "all rows, column 2", kept as a column: [[4]]. Writing out[:, 2] instead would give a flat list, and the division would pair things up wrongly once there are several points.


def reprojection_errors(H, pitch_xy, pixels):
    """(N,) distance in pixels between where H puts each point and where you clicked it."""
    projected = project(H, pitch_xy)
    errors = np.linalg.norm(projected - pixels, axis=1)
    return errors


def main(clicks_path):
    # Self-checks for project(): the toy H from idea 3, and OpenCV's own version.
    toy = np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]])
    assert np.allclose(project(toy, np.array([[2.0, 30.0]])), [[0.5, 7.5]]), "project() fails the toy example"

    names, pitch_xy, pixels = load_pairs(clicks_path)
    fit = np.array([n not in HELD_OUT for n in names])
    print(f"{fit.sum()} points to fit H, {(~fit).sum()} held out")

    H = compute_h(pitch_xy[fit], pixels[fit])
    cv_proj = cv2.perspectiveTransform(pitch_xy[None], H)[0]
    assert np.allclose(project(H, pitch_xy), cv_proj), "project() disagrees with cv2.perspectiveTransform"

    err = reprojection_errors(H, pitch_xy, pixels)
    for n, e, f in zip(names, err, fit):
        print(f"  {'fit     ' if f else 'HELD OUT'}  {n:25s} {e:6.2f} px")
    print(f"mean error  fit: {err[fit].mean():.2f} px   held out: {err[~fit].mean():.2f} px")
    np.set_printoptions(suppress=True, precision=4)
    print("H =\n", H / H[2, 2])

    # Leave-one-out: every point gets to be the held-out point once.
    loo = []
    for i in range(len(names)):
        keep = np.arange(len(names)) != i                     # True everywhere except point i
        H_i = compute_h(pitch_xy[keep], pixels[keep])         # H that never saw point i
        e_i = reprojection_errors(H_i, pitch_xy[i:i + 1], pixels[i:i + 1])[0]
        loo.append(e_i)
    print("\nleave-one-out (each point held out in turn):")
    for n, e in zip(names, loo):
        print(f"  {n:25s} {e:6.2f} px")
    print(f"mean leave-one-out error: {np.mean(loo):.2f} px   (checkpoint: < 5 px)")


if __name__ == "__main__":
    main(sys.argv[1])
