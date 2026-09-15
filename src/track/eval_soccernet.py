"""Stage 2: position error in meters on a SoccerNet GSR clip, against its ground-truth labels.

Usage: uv run python src/track/eval_soccernet.py SNGS-028
Reads data/soccernet/<clip>/Labels-GameState.json and data/camera/<clip>.json.

(A) camera only: SoccerNet's own (perfect) boxes -> foot pixel -> OUR H -> meters, vs the true
    meters. Detection and tracking play no part, so this is PnLCalib's error in meters.
(B) full pipeline (not yet): our boxes -> our tracks -> meters, matched to the true players.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from compare_pnl import GOAL_SIDE, pitch_to_soccernet  # noqa: E402  (lives in src/calib)
from to_pitch import pixels_to_meters  # noqa: E402

FPS = 25
PERSON = {1, 2, 3}  # SoccerNet category ids: player, goalkeeper, referee (4 = ball, 5 = pitch lines)


def load_truth(clip):
    """frame -> (foot_px (N, 2), true_xy (N, 2) SoccerNet meters, raw_xy (N, 2)) for every labelled person."""
    d = json.load(open(ROOT / "data" / "soccernet" / clip / "Labels-GameState.json"))
    frame_of = {im["image_id"]: int(im["file_name"].split(".")[0]) for im in d["images"]}  # 000001.jpg -> 1
    per_frame = {}
    for a in d["annotations"]:
        if a["category_id"] not in PERSON or a["bbox_pitch"] is None:
            continue  # a handful of people have no pitch position (off the pitch)
        b, p, r = a["bbox_image"], a["bbox_pitch"], a["bbox_pitch_raw"]
        foot = (b["x"] + b["w"] / 2, b["y"] + b["h"])  # middle of the bottom edge, like foot_point
        raw = (r["x_bottom_middle"], r["y_bottom_middle"]) if r else (np.nan, np.nan)  # raw is sometimes missing
        per_frame.setdefault(frame_of[a["image_id"]], []).append(
            (foot, (p["x_bottom_middle"], p["y_bottom_middle"]), raw))
    return {f: tuple(np.array(col, float) for col in zip(*rows)) for f, rows in per_frame.items()}


def camera_errors(H, foot_px, true_xy, goal):
    """Error in meters of each true foot pixel pushed through OUR camera.

    H: (3, 3) from camera.json, OUR pitch meters -> pixels.
    foot_px: (N, 2) true foot pixels [u, v] from the labels.
    true_xy: (N, 2) true positions in SoccerNet meters (X, Y): origin centre spot.
    goal: "left" / "right", for pitch_to_soccernet.
    Returns: (N,) distances in meters.
    Careful: our H gives OUR frame, the labels are in SoccerNet's. Both functions you need exist.
    """
    ours = pixels_to_meters(H, foot_px)  # (N, 2) OUR meters: origin = centre of the attacked goal line
    soccer = pitch_to_soccernet(ours, goal)[:, :2]  # (N, 2) SoccerNet X, Y (drop Z, it's 0 on the grass)
    return np.linalg.norm(soccer - true_xy, axis=1)  # straight-line distance per person


def summary(name, errs):
    errs = np.asarray(errs)
    return (f"{name:<14} mean {errs.mean():5.2f} m  median {np.median(errs):5.2f}  "
            f"95% {np.percentile(errs, 95):5.2f}  max {errs.max():6.2f}  ({len(errs)} people)")


def main(clip):
    goal = GOAL_SIDE[clip]
    truth = load_truth(clip)
    cams = {c["frame"]: np.array(c["H"]) for c in json.load(open(ROOT / "data" / "camera" / f"{clip}.json"))["frames"]}
    no_cam = sorted(set(truth) - set(cams))
    print(f"{clip}: {len(truth)} labelled frames, {len(no_cam)} without a PnLCalib camera (skipped)")

    # The labels' own wobble: SoccerNet stores two positions per person (bbox_pitch and bbox_pitch_raw).
    # How far apart they are = how exact the "truth" itself is. Don't expect to beat this.
    floor = np.concatenate([np.linalg.norm(t - r, axis=1) for _, t, r in truth.values()])
    print(summary("labels' own", floor[~np.isnan(floor)]))

    per_frame = {f: camera_errors(cams[f], foot, true, goal) for f, (foot, true, _) in truth.items() if f in cams}
    all_errs = np.concatenate(list(per_frame.values()))
    print(summary("(A) camera", all_errs))
    if np.median(all_errs) > 20:
        print("  median > 20 m: that's a mirrored or wrong-end pitch, check GOAL_SIDE / pitch_to_soccernet")

    print("(A) per 5 s:")
    for start in range(1, 751, 5 * FPS):
        chunk = [e for f, e in per_frame.items() if start <= f < start + 5 * FPS]
        if chunk:
            print(f"  frames {start:3d}-{start + 5 * FPS - 1:3d}  " + summary("", np.concatenate(chunk)).strip())
    worst = sorted(per_frame, key=lambda f: -per_frame[f].mean())[:5]
    print("(A) worst frames (mean m):", ", ".join(f"{f}: {per_frame[f].mean():.1f}" for f in worst))


if __name__ == "__main__":
    main(sys.argv[1])
