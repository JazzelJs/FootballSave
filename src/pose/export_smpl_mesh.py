"""Pack SMPL faces and animated vertices into one browser-readable binary."""

import argparse
import struct
from pathlib import Path

import numpy as np

from smpl import load_model


def export(source_path: Path, output_path: Path) -> None:
    source = np.load(source_path)
    frames = np.asarray(source["frame"], dtype="<i4")
    vertices = np.asarray(source["vertices"], dtype="<f4")
    faces = np.asarray(load_model().faces, dtype="<u4")
    fps = float(source["fps"])

    assert vertices.shape == (len(frames), 6890, 3), vertices.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape

    header = struct.pack("<4sIIIf", b"SMPL", len(frames), vertices.shape[1], len(faces), fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + frames.tobytes() + faces.tobytes() + vertices.tobytes())
    print(f"wrote {output_path} ({len(frames)} frames, {vertices.shape[1]} vertices, {len(faces)} faces)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("data/pose_166.npz"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/pose_166.smpl"))
    args = parser.parse_args()
    export(args.source, args.output)
