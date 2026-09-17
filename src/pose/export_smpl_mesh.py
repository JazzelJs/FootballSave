"""Pack SMPL faces and animated vertices into one browser-readable binary.

With --clip/--track the body is rotated out of HMR2's camera space into the viewer's pitch
space, so its facing is the real one and no per-clip flip is needed. Without them the vertices
are written as HMR2 produced them, which is only useful for looking at a pose on its own.
"""

import argparse
import struct
from pathlib import Path

import numpy as np

import orient
from smpl import load_model


def rotations_for(source, source_path, clip, track_id, frames, smooth_yaw):
    """Per-frame HMR2-camera -> scene rotation, with the yaw jitter smoothed out."""
    rotations = orient.scene_rotations(clip, track_id, frames)
    if smooth_yaw == 1:
        return rotations
    # apply_smpl.py drops the pose parameters when it bakes vertices, so read the facing back
    # from the NPZ they came from.
    if "global_orient_rotmat" not in source:
        source = np.load(source_path.with_name(source_path.name.replace("_vertices", "")))
    global_orient = np.asarray(source["global_orient_rotmat"], dtype=float)[:, 0]
    assert len(global_orient) == len(frames), (len(global_orient), len(frames))
    yaw = orient.scene_yaw(rotations, global_orient)
    return orient.yaw_smoothing(yaw, smooth_yaw) @ rotations


def export(source_path: Path, output_path: Path, clip: str | None = None,
           track_id: int | None = None, smooth_yaw: int = 1) -> None:
    source = np.load(source_path)
    frames = np.asarray(source["frame"], dtype="<i4")
    vertices = np.asarray(source["vertices"], dtype="<f4")
    faces = np.asarray(load_model().faces, dtype="<u4")
    fps = float(source["fps"])

    assert vertices.shape == (len(frames), 6890, 3), vertices.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape
    if clip is not None:
        rotations = rotations_for(source, source_path, clip, track_id, frames, smooth_yaw)
        vertices = np.einsum("nij,nvj->nvi", rotations, vertices).astype("<f4")
        assert np.all(np.linalg.det(rotations) > 0.99), "the body must be rotated, never mirrored"

    header = struct.pack("<4sIIIf", b"SMPL", len(frames), vertices.shape[1], len(faces), fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + frames.tobytes() + faces.tobytes() + vertices.tobytes())
    print(f"wrote {output_path} ({len(frames)} frames, {vertices.shape[1]} vertices, {len(faces)} faces)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("data/pose/npz/pose_166.npz"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/pose/mesh/pose_166.smpl"))
    parser.add_argument("--clip", help="rotate the body into pitch space with this clip's camera")
    parser.add_argument("--track", type=int, help="track id, for the crop centre each frame")
    parser.add_argument("--smooth-yaw", type=int, default=1,
                        help="odd moving-average window applied to the pitch-space yaw")
    args = parser.parse_args()
    if (args.clip is None) != (args.track is None):
        parser.error("--clip and --track go together")
    export(args.source, args.output, args.clip, args.track, args.smooth_yaw)
