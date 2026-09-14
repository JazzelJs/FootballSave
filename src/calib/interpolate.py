"""Moving camera: calibrate some frames by hand, guess H for the frames in between.

Test: click every 25th frame. Use every 50th (or 100th) as keyframes, interpolate H to the
clicked frames in between, and compare with your clicks there (frames H never saw).

Usage: uv run python src/calib/interpolate.py clip04
"""
import sys
from pathlib import Path

import numpy as np

from homography import compute_h, load_pairs, project, reprojection_errors
from pitch_model import POINTS

ROOT = Path(__file__).resolve().parents[2]
FPS = 49.95
GAPS = [50, 100]  # frames between keyframes (clicks are every 25th frame)


def interpolate_h(H_a, H_b, t):
    """Guess H for a frame between keyframe a (t = 0) and keyframe b (t = 1).

    H_a, H_b: 3x3, pitch meters -> pixels, as returned by compute_h.
    t: float from 0 to 1 (0.5 = halfway in time).
    Returns a 3x3 H. Use idea B (move the pixels, not the matrix numbers).
    Think about: does a point you use here have to be visible in both frames?
    """


    anchors = np.array([POINTS[n][:2] for n in POINTS])
    px_a = project(H_a, anchors)
    px_b = project(H_b, anchors)
    px_t = (1 - t) * px_a + t * px_b
    return compute_h(anchors, px_t)



def h_at(frame, H_by_frame):
    """Interpolated H for any frame between the first and last keyframe.

    frame: frame number, e.g. 25.
    H_by_frame: {keyframe number: H}, e.g. {0: H0, 50: H50, 100: H100}.
    Returns a 3x3 H (from interpolate_h, with the two keyframes around `frame`).
    """
    
    keyframes = sorted(H_by_frame.keys())
    for i in range(len(keyframes) - 1):
        if keyframes[i] <= frame <= keyframes[i + 1]:
            t = (frame - keyframes[i]) / (keyframes[i + 1] - keyframes[i])
            return interpolate_h(H_by_frame[keyframes[i]], H_by_frame[keyframes[i + 1]], t)
    raise ValueError(f"Frame {frame} is outside the range of keyframes {keyframes}.")


def main(clip):
    files = sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json"))
    clicks = {int(f.stem.split("_")[1]): load_pairs(f)[1:] for f in files}  # frame -> (pitch_xy, pixels)
    print(f"{clip}: clicked frames {sorted(clicks)}")

    # Self-checks for your two functions, on the first two clicked frames.
    (fa, (pa, xa)), (fb, (pb, xb)) = sorted(clicks.items())[:2]
    H_a, H_b = compute_h(pa, xa), compute_h(pb, xb)
    pts = np.array([POINTS[n][:2] for n in POINTS])
    same = lambda H1, H2: np.allclose(project(H1, pts), project(H2, pts), atol=0.01)
    assert same(interpolate_h(H_a, H_b, 0), H_a), "t = 0 should give keyframe a"
    assert same(interpolate_h(H_a, H_b, 1), H_b), "t = 1 should give keyframe b"
    assert same(interpolate_h(H_a, H_b, 0.5), interpolate_h(H_a, 1000 * H_b, 0.5)), \
        "1000 * H_b is the same camera as H_b, so the answer must not change"
    assert same(h_at(fa, {fa: H_a, fb: H_b}), H_a), "h_at on a keyframe should give that keyframe"

    for gap in GAPS:
        H_by_frame = {f: compute_h(p, x) for f, (p, x) in clicks.items() if f % gap == 0}
        tests = [f for f in sorted(clicks) if f not in H_by_frame and min(H_by_frame) < f < max(H_by_frame)]
        print(f"\nkeyframes every {gap} frames ({gap / FPS:.1f} s): {sorted(H_by_frame)}")
        print(f"  {'frame':>5}  {'interpolated':>12}  {'nearest keyframe':>16}   (mean / max px)")
        interp, nearest = [], []
        for f in tests:
            pitch_xy, pixels = clicks[f]
            key = min(H_by_frame, key=lambda k: abs(k - f))  # baseline: reuse the closest keyframe's H
            e_int = reprojection_errors(h_at(f, H_by_frame), pitch_xy, pixels)
            e_near = reprojection_errors(H_by_frame[key], pitch_xy, pixels)
            interp.append(e_int.mean())
            nearest.append(e_near.mean())
            print(f"  {f:5d}  {e_int.mean():5.1f} / {e_int.max():5.1f}  {e_near.mean():8.1f} / {e_near.max():5.1f}")
        print(f"  mean   {np.mean(interp):5.1f}          {np.mean(nearest):8.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
