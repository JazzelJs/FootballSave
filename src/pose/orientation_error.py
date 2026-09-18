"""Compare the SMPL facing direction with a tracked player's direction of travel.

Usage: uv run python src/pose/orientation_error.py SNGS-043 284

A running player faces where they run, so their motion is a usable stand-in for ground-truth
facing -- but only while they are actually running. Standing, turning and planted players face
anywhere, which is why --min-speed exists: below it the comparison measures nothing.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import orient

ROOT = Path(__file__).resolve().parents[2]
STEP = 2  # frames either side, so the direction is measured over 4 frames not tracker jitter


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def main(clip, track_id, smooth_window, min_speed):
    source = np.load(orient.pose_npz(clip, track_id))
    frames = np.asarray(source["frame"], dtype=int)
    global_orient = np.asarray(source["global_orient_rotmat"], dtype=float)[:, 0]
    rotations = orient.scene_rotations(clip, track_id, frames)
    yaw = orient.scene_yaw(rotations, global_orient)
    smoothed = orient.scene_yaw(orient.yaw_smoothing(yaw, smooth_window) @ rotations, global_orient)
    pose_yaw = dict(zip(frames, np.degrees(smoothed)))
    # What the old export did instead: yaw straight out of HMR2's camera, with one frame picked
    # by hand as the zero. Kept here so the camera-space and pitch-space numbers stand side by side.
    camera_forward = global_orient[:, :, 2]
    camera_yaw = dict(zip(frames, np.degrees(np.unwrap(np.arctan2(camera_forward[:, 0], camera_forward[:, 2])))))

    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    fps = tracks["fps"]
    positions = {f["frame"]: next((p for p in f["players"] if p["id"] == track_id), None)
                 for f in tracks["frames"]}
    rows = []
    for frame in frames:
        before, after = positions.get(frame - STEP), positions.get(frame + STEP)
        if not before or not after:
            continue
        dx, dz = after["x"] - before["x"], -(after["y"] - before["y"])
        speed = np.hypot(dx, dz) / (2 * STEP / fps)
        if speed < min_speed:
            continue
        run_yaw = np.degrees(np.arctan2(dx, dz))
        rows.append((int(frame), abs(wrap_degrees(pose_yaw[frame] - run_yaw)), camera_yaw[frame], run_yaw))
    if not rows:
        raise SystemExit(f"no samples above {min_speed} m/s in the pose window")
    errors = np.array([r[1] for r in rows])
    # The camera-space yaw has no zero of its own, so give it the best one it could have had:
    # the offset that minimises its own error. Anything worse than that it cannot blame on tuning.
    offsets = np.arange(-180, 180, 1.0)
    camera_errors = np.abs(wrap_degrees(np.array([r[2] for r in rows])[None].T + offsets
                                        - np.array([r[3] for r in rows])[None].T))
    best = camera_errors[:, np.argmin(np.median(camera_errors, axis=0))]
    print(f"{clip} track {track_id}: {len(rows)} samples above {min_speed} m/s "
          f"of {len(frames)} posed frames")
    print(f"  pitch-space facing : median {np.median(errors):5.1f}°  mean {np.mean(errors):5.1f}°  "
          f"95th {np.percentile(errors, 95):5.1f}°")
    print(f"  camera-space, best-case zero : median {np.median(best):5.1f}°  mean {np.mean(best):5.1f}°  "
          f"95th {np.percentile(best, 95):5.1f}°")
    print("  largest errors:", ", ".join(f"{f}={e:.0f}°" for f, e, _, _ in
          sorted(rows, key=lambda r: r[1], reverse=True)[:5]))
    assert abs(wrap_degrees(180.0)) == 180.0 and wrap_degrees(190.0) == -170.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("track_id", type=int)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-speed", type=float, default=3.0,
                        help="m/s below which a player is not reliably facing their direction of travel")
    args = parser.parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window must be an odd positive number")
    main(args.clip, args.track_id, args.smooth_window, args.min_speed)
