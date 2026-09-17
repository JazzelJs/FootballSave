"""D1: the depth-from-size baseline, on SoccerNet-v3D's own ground truth.

Usage: uv run python src/ball/size_baseline.py

The idea in one line: a ball is always 22 cm across, so if you know the focal length you can turn
its size in pixels into a distance --  distance ~= f * 0.22 / d_px.  It needs no physics, no
trajectory and no second camera: one box, one number. That is why it is the thing our physics fit
has to beat, and the paper says it is worth 4.2 m mean error.

Why it is fragile, in our own numbers: 1 px of box error costs 1.3 m at frame 602 and 6.9 m at
frame 620 of SNGS-043. The ball shrinks, the error per pixel grows.

WHAT IS IN THE FILE (measured 2026-09-18, use these instead of poking around):
  4051 rows, and `is_ball` is True in all of them -- so this file cannot tell you a detection
  rate, only an accuracy. Images are 1280x720.

  ball_bbox    "[198.16, 508.26, 26.76, 16.80]"  -> [x, y, w, h].
               Two things to settle before you trust it, and one is a trap:
                 - is (x, y) the top-left corner or the centre? You can SETTLE it rather than
                   guess: project ball_3D through calibration and see which one it lands on.
                 - a football is round, so why is this box 26.76 wide and 16.80 tall? One of
                   those two numbers is the ball's diameter and the other is not. Work out which,
                   and what is stretching the box in that direction. Getting this backwards
                   changes the answer by about 40%.
  calibration  a Python dict (single quotes -> ast.literal_eval, NOT json.loads) with exactly
               the keys PnLCalib gives us: pan/tilt/roll_degrees, x_focal_length, y_focal_length,
               principal_point, position_meters, rotation_matrix. So `camera_matrix(cam)` from
               src/calib/compare_pnl.py takes it unchanged -- do not write a second one.
  ball_3D      "[array([39.02, 2.3411, 0.0041256])]" -- a numpy repr inside a string, so it needs
               pulling apart with a regex or a str.strip chain. SoccerNet meters, Z pointing DOWN.
  optimized_d  their own refined distance for the same row, and `rep_error` / `optimized_error`
               are their reprojection errors. Free second opinion to compare yours against.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from compare_pnl import camera_matrix  # noqa: E402  reuse Stage 1's camera, do not rewrite it

CSV = ROOT / "data" / "snv3d" / "SNv3D.csv"
BALL_M = 0.22  # a size-5 football is 22 cm across


def rows():
    """Parse the csv into dicts with real types: bbox as 4 floats, calibration as a dict,
    ball_3D as 3 floats. Pure plumbing -- see the docstring above for each format."""
    # TODO(human)
    raise NotImplementedError


def distance_from_size(row):
    """The baseline: one number in, one distance out.

    distance ~= f * BALL_M / d_px, where f is the focal length in pixels and d_px is the ball's
    size in pixels. Decide which focal (there are two) and which side of the box, and say why.
    """
    # TODO(human)
    raise NotImplementedError


def true_distance(row):
    """How far the ball really is, from ball_3D and the camera's position_meters.

    This is the number to compare against, and it is a plain 3D distance between two points --
    no projection needed. Mind that ball_3D and position_meters are in the same SoccerNet frame.
    """
    # TODO(human)
    raise NotImplementedError


def main():
    data = rows()
    errors = [abs(distance_from_size(r) - true_distance(r)) for r in data]
    errors.sort()
    print(f"{len(data)} rows")
    print(f"depth-from-size error: mean {sum(errors) / len(errors):.2f} m, "
          f"median {errors[len(errors) // 2]:.2f} m, 95% {errors[int(len(errors) * 0.95)]:.2f} m")
    print("the paper reports 4.2 m mean for this method")


def demo():
    """Properties your answer must have. They do not tell you which box side or which focal."""
    data = rows()
    assert len(data) == 4051, len(data)

    # 1. The camera is a real camera: Stage 1's camera_matrix must accept the calibration dict
    #    unchanged, and a 3x4 P must come out.
    assert camera_matrix(data[0]["calibration"]).shape == (3, 4)

    # 2. Sanity on the scale: a broadcast ball is tens of meters away, never 2 m and never 500 m.
    #    If your distance is out by a factor of ~1.6 you have picked the wrong side of the box.
    for row in data[:200]:
        d = distance_from_size(row)
        assert 5.0 < d < 200.0, f"{d:.1f} m is not a broadcast distance"

    # 3. The real test: does this beat, match, or lose to the paper's 4.2 m mean? Any of the three
    #    is a valid result -- but a mean UNDER ~2 m means you are accidentally reading their
    #    answer (optimized_d) rather than computing your own.
    errors = [abs(distance_from_size(r) - true_distance(r)) for r in data]
    mean = sum(errors) / len(errors)
    assert 1.0 < mean < 20.0, f"mean {mean:.2f} m -- too good or too bad to be this method"
    print(f"size_baseline self-check ok: mean {mean:.2f} m over {len(data)} rows")


if __name__ == "__main__":
    demo() if "--self-check" in sys.argv else main()
