"""Test PnLCalib's cameras against your clicks, next to interpolation and the own-fit floor.

Needs data/pnlcalib/pnlcalib_raw_<clip>.json (output of notebooks/kaggle/pnlcalib_clip04.ipynb).
Usage: uv run python src/calib/compare_pnl.py clip04
"""
import json
import sys
from pathlib import Path

import numpy as np

import interpolate
from homography import compute_h, load_pairs, project, reprojection_errors
from pitch_model import POINTS

ROOT = Path(__file__).resolve().parents[2]


def camera_matrix(cam):
    """PnLCalib cam_params -> 3x4 P: SoccerNet meters (X, Y, Z, 1) -> pixels.
    Same as projection_from_cam_params() in PnLCalib's inference.py: P = K R [I | -C]."""
    K = np.array([[cam["x_focal_length"], 0, cam["principal_point"][0]],
                  [0, cam["y_focal_length"], cam["principal_point"][1]],
                  [0, 0, 1]])
    R = np.array(cam["rotation_matrix"])
    C = np.array(cam["position_meters"])  # where the camera stands
    return K @ R @ np.hstack([np.eye(3), -C[:, None]])


def project_3d(P, xyz):
    """(N,3) meters -> (N,2) pixels with a 3x4 camera P. Same idea as project() in homography.py."""
    out = np.hstack([xyz, np.ones((len(xyz), 1))]) @ P.T
    return out[:, :2] / out[:, 2:3]


def pitch_to_soccernet(xy):
    """Points on the grass: our pitch meters -> SoccerNet meters.

    xy: (N, 2) in our frame: origin = centre of the attacked goal line, x = goalkeeper's right,
        y = into the field.
    Returns (N, 3) SoccerNet (X, Y, Z): origin = centre spot, X along the pitch (goal lines at
        X = -52.5 and +52.5), Y along the goal lines, Z pointing DOWN (grass = 0).
    Think about: which end is our goal, and which way does our +x point? Two yes/no choices.
    A wrong choice mirrors the pitch; the error table will show it.
    """
    # ponytail: clip04 only (goal on the LEFT of the screen). Goal on the right: X = 52.5 - y, Y = -x.
    origin = np.array([-52.5, 0.0, 0.0])  # OUR origin (centre of the attacked goal line), in SoccerNet meters
    soccer_xy = np.zeros((xy.shape[0], 3))
    soccer_xy[:, 0] = origin[0] + xy[:, 1]  # into the field = towards the centre spot = +X
    soccer_xy[:, 1] = origin[1] + xy[:, 0]  # goalkeeper's right = near side, where the camera stands = +Y
    soccer_xy[:, 2] = 0.0  # Z = 0 on the grass
    return soccer_xy


def camera_to_h(P):
    """Full camera -> H for the grass, in OUR pitch frame.

    P: 3x4, SoccerNet meters (X, Y, Z, 1) -> pixels.
    Returns 3x3 H: our pitch meters (x, y) -> pixels, the same kind of H as compute_h gives.
    Think about: what do all points on the grass have in common?
    """
    # Any 4 grass points pin down an H: take the pitch corners (our frame: x = ±34, y = 0 or 105),
    # see where the full camera puts them, and fit the H through those pixels.
    # (Shortcut: grass has Z = 0, so P's Z column never counts: P[:, [0, 1, 3]] is H in SoccerNet meters.)
    anchors = np.array([[-34.0, 0.0], [34.0, 0.0], [-34.0, 105.0], [34.0, 105.0]])
    H = compute_h(anchors, project_3d(P, pitch_to_soccernet(anchors)))
    return H


def main(clip):
    raw = json.load(open(ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json"))
    P_by_frame = {r["frame"]: camera_matrix(r["cam_params"]) for r in raw["frames"] if r["ok"]}
    files = sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json"))
    clicks = {int(f.stem.split("_")[1]): load_pairs(f)[1:] for f in files}  # frame -> (pitch_xy, pixels)

    # Self-checks for your two functions.
    g = pitch_to_soccernet(np.array([[0.0, 0.0], [0.0, 11.0]]))  # our origin, our penalty spot
    assert g.shape == (2, 3) and np.allclose(g[:, 2], 0), "return (N, 3), with Z = 0 on the grass"
    assert np.isclose(abs(g[0, 0]), 52.5) and np.isclose(g[0, 1], 0), "our origin is the centre of a goal line"
    assert np.isclose(abs(g[1, 0]), 41.5) and np.isclose(g[1, 1], 0), "the penalty spot is 11 m towards the centre"
    pts = np.array([POINTS[n][:2] for n in POINTS])
    P0 = P_by_frame[min(P_by_frame)]
    assert np.allclose(project(camera_to_h(P0), pts), project_3d(P0, pitch_to_soccernet(pts)), atol=0.5), \
        "on the grass, your H must put every point where the full camera puts it"

    H50 = {f: compute_h(p, x) for f, (p, x) in clicks.items() if f % 50 == 0}
    print(f"{clip}: error vs your clicks (mean / max px)")
    print(f"  {'frame':>5}  {'own fit':>13}  {'PnLCalib':>13}  {'interp (1 s)':>13}")
    rows = []
    for f in sorted(clicks):
        pitch_xy, pixels = clicks[f]
        own = reprojection_errors(compute_h(pitch_xy, pixels), pitch_xy, pixels)
        pnl = reprojection_errors(camera_to_h(P_by_frame[f]), pitch_xy, pixels)
        cols = [own, pnl]
        if f not in H50 and min(H50) < f < max(H50):
            cols.append(reprojection_errors(interpolate.h_at(f, H50), pitch_xy, pixels))
        print(f"  {f:5d}  " + "  ".join(f"{e.mean():5.1f} / {e.max():5.1f}" for e in cols))
        rows.append((own.mean(), pnl.mean(), cols[2].mean() if len(cols) == 3 else np.nan))
    own_m, pnl_m, int_m = np.nanmean(rows, axis=0)
    print(f"  {'mean':>5}  {own_m:5.1f}          {pnl_m:5.1f}          {int_m:5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
