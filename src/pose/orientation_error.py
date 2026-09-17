"""Compare the adjusted SMPL facing direction with a tracked player's motion.

Usage: uv run python src/pose/orientation_error.py SNGS-043 1131 --reference-frame 602
This is a proxy: it measures facing vs movement, not ground-truth body facing.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def main(clip, track_id, reference_frame, smooth_window):
    source = np.load(ROOT / "data" / "pose" / "npz" / f"pose_{track_id}.npz")
    frames = np.asarray(source["frame"], dtype=int)
    forward = np.asarray(source["global_orient_rotmat"], dtype=float)[:, 0] @ np.array([0, 0, 1.0])
    raw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    pad = smooth_window // 2
    smooth = np.convolve(np.pad(raw, pad, mode="edge"),
                         np.ones(smooth_window) / smooth_window, mode="valid")
    ref = np.flatnonzero(frames == reference_frame)
    if len(ref) != 1:
        raise SystemExit(f"reference frame {reference_frame} is not present exactly once")
    # The exported mesh mirrors X, so its pitch-world yaw is the negative of the aligned yaw.
    facing = -(smooth - smooth[ref[0]])
    pose_yaw = dict(zip(frames, np.degrees(facing)))

    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    positions = {f["frame"]: next((p for p in f["players"] if p["id"] == track_id), None)
                 for f in tracks["frames"]}
    rows = []
    for frame in frames:
        before, after = positions.get(frame - 2), positions.get(frame + 2)
        if not before or not after:
            continue
        dx, dz = after["x"] - before["x"], -(after["y"] - before["y"])
        if np.hypot(dx, dz) < 0.1:
            continue
        run_yaw = np.degrees(np.arctan2(dx, dz))
        rows.append((frame, float(pose_yaw[frame]), float(run_yaw),
                     abs(wrap_degrees(pose_yaw[frame] - run_yaw))))
    if not rows:
        raise SystemExit("no moving track samples in the pose window")
    errors = np.array([r[3] for r in rows])
    print(f"{clip} track {track_id}: {len(rows)} moving samples")
    print(f"facing-vs-running angle: median {np.median(errors):.1f}°, "
          f"mean {np.mean(errors):.1f}°, 95th percentile {np.percentile(errors, 95):.1f}°")
    print("largest errors:", ", ".join(f"{f}={e:.1f}°" for f, _, _, e in
          sorted(rows, key=lambda r: r[3], reverse=True)[:5]))
    assert abs(wrap_degrees(180.0)) == 180.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("track_id", type=int)
    parser.add_argument("--reference-frame", type=int, default=602)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window must be an odd positive number")
    main(args.clip, args.track_id, args.reference_frame, args.smooth_window)
