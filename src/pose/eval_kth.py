"""Score HMR2 joints against KTH Football II ground truth.

Usage: uv run python src/pose/eval_kth.py

PA-MPJPE aligns each predicted frame to its KTH frame with one similarity
transform (rotation, scale and translation), then averages its 14 joint errors.
Facing uses one transform for the whole sequence so frame-to-frame direction is
not erased by the per-frame PA alignment.
"""
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KTH_TO_SMPL = np.array([8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15])
RIGHT_HIP, LEFT_HIP = 2, 3


def similarity(predicted, target):
    """Return `predicted` after its least-squares rotation, scale and shift to target."""
    predicted_center = predicted.mean(axis=0)
    target_center = target.mean(axis=0)
    left = predicted - predicted_center
    right = target - target_center
    u, singular, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = singular.sum() / np.square(left).sum()
    return scale * left @ rotation + target_center


def facing(points):
    """Horizontal normal of the hip line; KTH's z coordinate is vertical."""
    hip_line = points[LEFT_HIP] - points[RIGHT_HIP]
    forward = np.cross(hip_line, [0.0, 0.0, 1.0])[:2]
    return forward / np.linalg.norm(forward)


def angle_degrees(first, second):
    return np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


def main():
    predicted = np.load(ROOT / "data/models/kth_camera1_pose.npz")["joints"][:, KTH_TO_SMPL]
    target = np.loadtxt(ROOT / "data/kth/sequence2/positions3d.txt").reshape(-1, 14, 3)
    assert predicted.shape == target.shape == (175, 14, 3)

    pa_errors = np.array([
        np.linalg.norm(similarity(frame, truth) - truth, axis=1).mean()
        for frame, truth in zip(predicted, target)
    ])
    # One transform preserves the model's changes in facing across the sequence.
    sequence_aligned = similarity(predicted.reshape(-1, 3), target.reshape(-1, 3)).reshape(predicted.shape)
    facing_errors = np.array([
        angle_degrees(facing(frame), facing(truth))
        for frame, truth in zip(sequence_aligned, target)
    ])

    print(f"KTH sequence 2, Camera 1: {len(target)} frames, 14 joints")
    print(f"PA-MPJPE: {pa_errors.mean() * 1000:.1f} mm mean, {np.median(pa_errors) * 1000:.1f} mm median")
    print(f"facing error (one sequence alignment): {facing_errors.mean():.1f}° mean, "
          f"{np.median(facing_errors):.1f}° median, {np.percentile(facing_errors, 95):.1f}° 95th percentile")
    assert np.isclose(angle_degrees([1, 0], [0, 1]), 90.0)


if __name__ == "__main__":
    main()
