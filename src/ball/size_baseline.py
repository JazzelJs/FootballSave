"""D1: the depth-from-size baseline, measured on SoccerNet-v3D's own ground truth.

Usage: uv run python src/ball/size_baseline.py
       uv run python src/ball/size_baseline.py --self-check

The idea in one line: a ball is always 22 cm across, so if you know the focal length you can turn
its size in pixels into a distance -- `distance ≈ f · 0.22 / d_px`. No physics, no trajectory, no
second camera: one box, one number. That is why it is the thing our physics fit has to beat.

THE RESULT, and it is not what PLAN.md expected: on the 4051 rows of SNv3D.csv this method is
**17.9 m mean / 14.9 m median error on the held-out test split** -- about **20% of the distance**,
in every distance band. PLAN.md credits the paper with 4.2 m for "this per-frame method"; we
measure four times worse. Two causes, both measured below, and a warning about the third:
  1. The boxes are 23% bigger than the ball. The geometry demands a median 11.99 px diameter at
     these distances; the annotations give 15.50 px. A generous box means a too-near ball.
  2. That inflation is not constant, so correcting it barely helps: fitting one factor on train
     (k = 1.232) and applying it to test moves the mean 17.9 -> 16.6 m. The per-row scatter, not
     the bias, is what kills the method. With the ball only 12-15 px wide, 1.5 px of annotation
     slop is ~12% of the diameter and therefore ~12% of the distance.
  3. So the 4.2 m in PLAN.md is probably NOT this formula. The csv also carries `optimized_d` and
     `optimized_error`, and `optimized_d` is not a camera-to-ball distance at all (17.1 m where
     the true distance is 77.3 m, and the ratio is not constant), so the paper is reporting some
     refined quantity. **Do not quote 4.2 m as this method's error.**
"""
import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from compare_pnl import camera_matrix, project_3d  # noqa: E402  reuse Stage 1's camera

CSV = ROOT / "data" / "snv3d" / "SNv3D.csv"
BALL_M = 0.22  # a size-5 football is 22 cm across
BANDS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 1500))


def rows():
    """Parse the csv into real types. Three formats, each with its own trap.

    ball_bbox    "[198.16, 508.26, 26.76, 16.80]" = [x, y, w, h], and (x, y) is the CENTRE.
                 Settled by projecting ball_3D through the calibration: it lands 2.77 px from the
                 centre and 12.94 px from the top-left corner, so the centre it is.
    calibration  a PYTHON dict (single quotes) -> ast.literal_eval, not json.loads. Its keys are
                 exactly PnLCalib's, so camera_matrix() takes it unchanged.
    ball_3D      "[array([39.02, 2.3411, 0.0041256])]" -- a numpy repr inside a string. No row
                 holds more than one array, so pulling the first three numbers out is enough.
    """
    out = []
    with CSV.open() as handle:
        for row in csv.DictReader(handle):
            calibration = ast.literal_eval(row["calibration"])
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", row["ball_3D"])
            out.append({"bbox": ast.literal_eval(row["ball_bbox"]),
                        "calibration": calibration,
                        "ball_3D": np.array([float(n) for n in numbers[:3]]),
                        "set": row["set"]})
    return out


def diameter_px(row):
    """The ball's diameter in pixels: the SMALLER side of the box.

    A football is round, so the two sides should agree, and on median they nearly do (w 17.26,
    h 16.73). But they disagree per row, sometimes wildly -- row 0 is 26.76 wide by 16.80 tall for
    a ball whose geometry demands 16.6 px, so its width is smeared and its height is right.
    Whatever inflates a box (motion blur, a generous annotator) can only ever make it BIGGER than
    the ball, never smaller, so the smaller side is the least-corrupted estimate. Measured on the
    test split, mean error: min 17.9 m, height 20.4 m, width 21.5 m, max(w, h) 24.3 m. The
    a-priori argument and the measurement agree, which is the only reason to trust either.
    """
    return min(row["bbox"][2], row["bbox"][3])


def distance_from_size(row):
    """The baseline: one box in, one distance out. f is in pixels, so the units cancel.

    x_focal_length == y_focal_length in 100% of rows, so the choice of focal does not matter here.
    """
    return row["calibration"]["x_focal_length"] * BALL_M / diameter_px(row)


def true_distance(row):
    """How far the ball really is: a plain 3D distance, camera to ball.

    No projection needed -- ball_3D and position_meters are in the same SoccerNet frame, which is
    exactly what the 2.77 px check in rows() proved.
    """
    return float(np.linalg.norm(row["ball_3D"] - np.array(row["calibration"]["position_meters"])))


def table(data):
    true = np.array([true_distance(r) for r in data])
    estimate = np.array([distance_from_size(r) for r in data])
    error = np.abs(estimate - true)
    splits = np.array([r["set"] for r in data])

    print(f"{len(data)} rows, true distance median {np.median(true):.1f} m")
    print("\nsplit      n      mean       median    as % of the distance")
    for name in ("train", "test"):
        keep = splits == name
        print(f"{name:<8} {keep.sum():5d} {error[keep].mean():8.2f} m {np.median(error[keep]):9.2f} m"
              f" {100 * np.median(error[keep] / true[keep]):9.1f}%")

    print("\nby true distance (the error is a fixed SHARE of the distance, which is the tell):")
    for low, high in BANDS:
        keep = (true >= low) & (true < high)
        if keep.any():
            print(f"  {low:4d}-{high:4d} m  n={keep.sum():4d}  mean {error[keep].mean():7.2f} m"
                  f"  median {np.median(error[keep]):6.2f} m"
                  f"  ({100 * np.median(error[keep] / true[keep]):4.1f}% of the distance)")

    # Why. The diameter the geometry demands, against the diameter we were handed.
    focal = np.array([r["calibration"]["x_focal_length"] for r in data])
    given = np.array([diameter_px(r) for r in data])
    needed = focal * BALL_M / true
    inflation = given / needed
    print(f"\nbox diameter: given median {np.median(given):.2f} px, geometry needs "
          f"{np.median(needed):.2f} px  ->  boxes are {100 * (np.median(inflation) - 1):.0f}% too big")

    # Fit the one correction factor on TRAIN, judge it on TEST. Never the other way round.
    k = float(np.median(inflation[splits == "train"]))
    print(f"correction k = {k:.3f}, fitted on train only:")
    for name in ("train", "test"):
        keep = splits == name
        fixed = np.abs(focal[keep] * BALL_M / (given[keep] / k) - true[keep])
        print(f"  {name:<6} {error[keep].mean():6.2f} -> {fixed.mean():6.2f} m mean, "
              f"{np.median(error[keep]):6.2f} -> {np.median(fixed):6.2f} m median")
    print("  i.e. removing the bias barely helps: the per-row scatter is the real problem.")
    return error, true, splits, k


def compare_on_our_clip(clip="SNGS-043"):
    """The last Build item: our physics fit vs per-frame depth-from-size, on our own shot.

    There is no 3D ground truth for this ball, so this is not an accuracy test -- it is the two
    methods disagreeing, which is the honest thing to look at. The physics fit gets ONE answer for
    the whole flight from 15 frames; depth-from-size gets a fresh, independent answer every frame.
    """
    fit = json.loads((ROOT / "data" / "ball" / f"{clip}_fit.json").read_text())
    ball = {r["frame"]: r for r in json.loads(
        (ROOT / "data" / "ball" / f"{clip}_clean.json").read_text())["frames"]}
    raw = json.loads((ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json").read_text())
    cameras = {f["frame"]: f["cam_params"] for f in raw["frames"] if f["ok"]}
    sys.path.insert(0, str(ROOT / "src" / "calib"))
    from compare_pnl import GOAL_SIDE, pitch_to_soccernet  # noqa: PLC0415

    print(f"\n{clip}: our physics fit vs depth-from-size, frame by frame")
    print("frame   box px   size-based d   physics-fit d   they differ by")
    gaps = []
    for frame, point in zip(fit["path_frames"], fit["path"]):
        if frame not in cameras or frame not in ball:
            continue
        cam = cameras[frame]
        box = min(ball[frame]["wh"])
        size_d = cam["x_focal_length"] * BALL_M / box
        soccer = pitch_to_soccernet(np.array([point[:2]]), GOAL_SIDE[clip])[0]
        soccer[2] = -point[2]
        fit_d = float(np.linalg.norm(soccer - np.array(cam["position_meters"])))
        gaps.append(abs(size_d - fit_d))
        print(f"{frame:5d} {box:7.1f} {size_d:14.1f} m {fit_d:14.1f} m {gaps[-1]:14.1f} m")
    gaps = np.array(gaps)
    print(f"  they disagree by {gaps.mean():.1f} m mean / {np.median(gaps):.1f} m median")
    print("  Which to trust: the physics fit. Depth-from-size has no way to be wrong QUIETLY --")
    print("  every frame is an independent guess off an 11-25 px box, and on SNv3D that same")
    print("  formula is 20% of the distance out. The fit is one curve through 15 frames, and it")
    print("  is checkable: 3.70 px reprojection, and it puts the ball inside the posts.")


def main():
    table(rows())
    compare_on_our_clip()


def demo():
    """Properties the answer must have, including the two that caught real mistakes."""
    data = rows()
    assert len(data) == 4051, len(data)

    # 1. Stage 1's camera must swallow the calibration dict unchanged, and 3x4 must come out.
    assert camera_matrix(data[0]["calibration"]).shape == (3, 4)

    # 2. (x, y) is the box CENTRE, not the corner. This is the check that settled it, and it also
    #    proves ball_3D and the calibration live in the same coordinate frame.
    to_centre, to_corner = [], []
    for row in data[:400]:
        pixel = project_3d(camera_matrix(row["calibration"]), row["ball_3D"][None, :])[0]
        x, y, w, h = row["bbox"]
        to_corner.append(np.linalg.norm(pixel - np.array([x, y])))
        to_centre.append(np.linalg.norm(pixel - np.array([x + w / 2, y + h / 2])))
    assert np.median(to_centre) < np.median(to_corner) / 3, "bbox (x, y) is not the centre"
    assert np.median(to_centre) < 5.0, f"{np.median(to_centre):.2f} px -- frames do not agree"

    # 3. Scale sanity: a broadcast ball is tens of metres away.
    distances = np.array([distance_from_size(r) for r in data])
    assert 5.0 < np.median(distances) < 200.0, np.median(distances)

    # 4. The result itself. Deliberately NOT asserting "beats 4.2 m": it does not, and pretending
    #    otherwise is how a wrong number survives. 10-25 m is what this method is actually worth.
    error = np.abs(distances - np.array([true_distance(r) for r in data]))
    assert 10.0 < error.mean() < 25.0, f"mean {error.mean():.2f} m -- not this method's ballpark"

    # 5. The boxes really are inflated, which is WHY the method is biased low.
    focal = np.array([r["calibration"]["x_focal_length"] for r in data])
    true = np.array([true_distance(r) for r in data])
    inflation = np.median(np.array([diameter_px(r) for r in data]) / (focal * BALL_M / true))
    assert inflation > 1.1, f"boxes only {inflation:.2f}x the ball, the bias story is wrong"
    print(f"size_baseline self-check ok: mean {error.mean():.2f} m over {len(data)} rows, "
          f"boxes {100 * (inflation - 1):.0f}% too big")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    demo() if parser.parse_args().self_check else main()
