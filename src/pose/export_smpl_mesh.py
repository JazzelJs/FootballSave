"""Pack SMPL faces and animated vertices into one browser-readable binary."""

import argparse
import struct
from pathlib import Path

import numpy as np

from smpl import load_model


def export(source_path: Path, output_path: Path, stabilize_yaw: bool = False,
           mirror_left_right: bool = False, align_yaw_frame: int | None = None,
           smooth_yaw: int = 1) -> None:
    source = np.load(source_path)
    frames = np.asarray(source["frame"], dtype="<i4")
    vertices = np.asarray(source["vertices"], dtype="<f4")
    faces = np.asarray(load_model().faces, dtype="<u4")
    fps = float(source["fps"])

    assert vertices.shape == (len(frames), 6890, 3), vertices.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape
    if stabilize_yaw or align_yaw_frame is not None:
        rotations = np.asarray(source["global_orient_rotmat"], dtype="<f4")[:, 0]
        forward = rotations @ np.array([0, 0, 1], dtype=np.float32)
        raw_yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
        if stabilize_yaw:
            angle = -raw_yaw
        else:
            if smooth_yaw < 1 or smooth_yaw % 2 == 0:
                raise ValueError("--smooth-yaw must be an odd positive number")
            pad = smooth_yaw // 2
            smooth = np.convolve(np.pad(raw_yaw, pad, mode="edge"),
                                 np.ones(smooth_yaw) / smooth_yaw, mode="valid")
            ref = np.flatnonzero(frames == align_yaw_frame)
            if len(ref) != 1:
                raise ValueError(f"alignment frame {align_yaw_frame} is not present exactly once")
            # Remove frame-to-frame yaw jitter but preserve the turn visible in HMR2's preview.
            angle = smooth - raw_yaw - smooth[ref[0]]
        c, s = np.cos(angle), np.sin(angle)
        yaw = np.zeros_like(rotations)
        yaw[:, 0, 0], yaw[:, 0, 2] = c, s
        yaw[:, 1, 1] = 1
        yaw[:, 2, 0], yaw[:, 2, 2] = -s, c
        vertices = np.einsum("nij,nvj->nvi", yaw, vertices)
        corrected = np.einsum("nij,nj->ni", yaw, forward)
        if stabilize_yaw:
            assert np.allclose(corrected[:, 0], 0, atol=1e-5) and np.all(corrected[:, 2] > 0), \
                "stabilized body must face +z"
        else:
            assert abs(corrected[ref[0], 0]) < 1e-5 and corrected[ref[0], 2] > 0, \
                "alignment frame must face +z"
    if mirror_left_right:
        vertices = vertices.copy()
        before = vertices[:, :, 0].copy()
        vertices[:, :, 0] *= -1
        assert np.array_equal(vertices[:, :, 0], -before), "left/right mirror must negate only x"

    header = struct.pack("<4sIIIf", b"SMPL", len(frames), vertices.shape[1], len(faces), fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + frames.tobytes() + faces.tobytes() + vertices.tobytes())
    print(f"wrote {output_path} ({len(frames)} frames, {vertices.shape[1]} vertices, {len(faces)} faces)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("data/pose/npz/pose_166.npz"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/pose/mesh/pose_166.smpl"))
    parser.add_argument("--stabilize-yaw", action="store_true",
                        help="preserve pose tilt but make every frame face the viewer's +z goal direction")
    parser.add_argument("--mirror-left-right", action="store_true",
                        help="mirror the body when HMR2 assigns an action to the wrong side")
    parser.add_argument("--align-yaw-frame", type=int,
                        help="preserve the preview's body turn but align this frame toward +z")
    parser.add_argument("--smooth-yaw", type=int, default=1,
                        help="odd moving-average window used with --align-yaw-frame")
    args = parser.parse_args()
    export(args.source, args.output, args.stabilize_yaw, args.mirror_left_right,
           args.align_yaw_frame, args.smooth_yaw)
