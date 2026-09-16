"""Write camera.json for Stage 2: one H per frame (our pitch meters -> pixels), from PnLCalib.

Usage: uv run python src/calib/export_camera.py clip04
Reads data/pnlcalib/pnlcalib_raw_<clip>.json, writes data/camera/<clip>.json.

Frames where PnLCalib found no camera, or a broken one (see broken()), get an H filled in from the
good frames around them with h_at from Stage 1 ("filled": true). Frames before the first or after
the last good frame have nothing on one side: they are left out.
"""
import json
import sys
from pathlib import Path

import numpy as np

from compare_pnl import GOAL_SIDE, camera_matrix, camera_to_h
from homography import load_pairs, project, reprojection_errors
from interpolate import h_at, interpolate_h

ROOT = Path(__file__).resolve().parents[2]
MAX_REP_ERR = 10     # px. PnLCalib's own fit error is ~4 px (95% under 5.3): above 10 it couldn't fit the lines
MIN_FOCAL = 1000     # px. Below this the view is wider than ~90 degrees: no broadcast camera zooms out that far
MAX_JUMP = 5         # m. How far this frame's camera may disagree with the neighbouring frames' cameras
GRID = np.array([[u, v] for u in (240, 960, 1680) for v in (600, 840, 1080)], float)  # spots in the lower picture


def broken(r):
    """A PnLCalib result we don't trust. On SNGS-028/043 these frames threw feet 10-3,000,000 m off."""
    return not r["ok"] or r["rep_err_px"] > MAX_REP_ERR or r["cam_params"]["x_focal_length"] < MIN_FOCAL


def jumpy(good):
    """Frames whose camera disagrees with the frames either side of it, by more than MAX_JUMP meters.

    broken() misses these: PnLCalib sometimes fits the few visible lines beautifully with a camera that
    stands 25 m away from where it really is (SNGS-043 frame 357: its own error is 0.4 px, the best of
    its neighbourhood, but players land 3.7 m off). One frame can be wrong like that; the frames before
    and after it can't be wrong in the same way at the same time.
    Measured in meters on the grass: where a player standing under each GRID pixel would be, this
    frame's camera vs the neighbours' interpolated one. A normal frame disagrees by 0.2-0.3 m.
    ponytail: one bad frame also makes its two neighbours disagree, so 3 frames get filled instead
    of 1. Harmless (they are filled from good frames either side); flag one at a time if it matters.
    """
    fs = sorted(good)
    out = set()
    for a, f, b in zip(fs, fs[1:], fs[2:]):
        guess = interpolate_h(good[a], good[b], (f - a) / (b - a))
        moved = np.linalg.norm(project(np.linalg.inv(good[f]), GRID) - project(np.linalg.inv(guess), GRID), axis=1)
        if np.median(moved) > MAX_JUMP:
            out.add(f)
    return out


def main(clip):
    goal = GOAL_SIDE[clip]  # KeyError = new clip: add its goal side in compare_pnl.py first
    raw = json.load(open(ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json"))

    good = {r["frame"]: camera_to_h(camera_matrix(r["cam_params"]), goal)  # your function does the real work
            for r in raw["frames"] if not broken(r)}
    jumped = jumpy(good)
    good = {f: H for f, H in good.items() if f not in jumped}
    frames = []
    for r in raw["frames"]:
        f = r["frame"]
        if f in good:
            H, filled = good[f], False
        elif min(good) < f < max(good):
            H, filled = h_at(f, good), True  # blend the pixels of the good frames before and after
        else:
            continue  # no good frame on one side: leave it out, Stage 2 skips it
        frames.append({"frame": f, "H": (H / H[2, 2]).tolist(), "pnl_rep_err_px": r["rep_err_px"] if r["ok"] else None,
                       "filled": filled})
    bad = [r["frame"] for r in raw["frames"] if broken(r)]
    print(f"{len(bad)} broken or missing PnLCalib frames, filled from neighbours: {bad}")
    print(f"{len(jumped)} more disagree with their neighbours by over {MAX_JUMP} m, also filled: {sorted(jumped)}")

    out = ROOT / "data" / "camera" / f"{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "clip": clip,
        "image_size": raw["image_size"],
        "H": "pitch meters (x, y, 1) -> pixels (u, v, w); pixels -> meters = inv(H)",
        "source": raw["method"] + f", converted with compare_pnl.pitch_to_soccernet (goal {goal})",
        "frames": frames,
    }, indent=1) + "\n")
    print(f"{len(frames)}/{len(raw['frames'])} frames -> {out.relative_to(ROOT)}")

    # Check: read the file back and test it on your clicks (should match compare_pnl's 5.9 px).
    H_by_frame = {f["frame"]: np.array(f["H"]) for f in json.load(open(out))["frames"]}
    errs = [reprojection_errors(H_by_frame[int(p.stem.split("_")[1])], *load_pairs(p)[1:]).mean()
            for p in sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json"))]
    if errs:  # SoccerNet clips have no clicks: they get checked against their labels instead
        print(f"read back: mean error on {len(errs)} clicked frames = {np.mean(errs):.1f} px")


if __name__ == "__main__":
    main(sys.argv[1])
