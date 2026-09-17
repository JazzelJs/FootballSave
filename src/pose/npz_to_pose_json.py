"""Convert the Stage 4 joints from NumPy NPZ to small browser-readable JSON."""

import ast
import json
import struct
import sys
import zipfile
from pathlib import Path


def read_npy(raw: bytes):
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an NPY array")
    major, minor = raw[6], raw[7]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        start = 10
    else:
        header_len = struct.unpack("<I", raw[8:12])[0]
        start = 12
    header = ast.literal_eval(raw[start : start + header_len].decode("latin1"))
    if header["descr"] != "<f4" and header["descr"] != "<i8":
        raise ValueError(f"unsupported dtype: {header['descr']}")
    shape = header["shape"] if isinstance(header["shape"], tuple) else (header["shape"],)
    count = 1
    for size in shape:
        count *= size
    offset = start + header_len
    if header["descr"] == "<f4":
        return list(struct.unpack(f"<{count}f", raw[offset : offset + count * 4])), shape
    return list(struct.unpack(f"<{count}q", raw[offset : offset + count * 8])), shape


def convert(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        frames, frame_shape = read_npy(archive.read("frame.npy"))
        joints, joint_shape = read_npy(archive.read("joints.npy"))
    assert frame_shape == (60,), frame_shape
    assert joint_shape[1:] == (24, 3), joint_shape
    per_frame = [joints[i * 24 * 3 : (i + 1) * 24 * 3] for i in range(joint_shape[0])]
    pose = {
        "frames": [
            {"frame": int(frame), "joints": [row[i * 3 : i * 3 + 3] for i in range(24)]}
            for frame, row in zip(frames, per_frame)
        ]
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pose, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/pose/npz/pose_166.npz")
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else source.parents[1] / "json" / f"{source.stem}.json"
    convert(source, destination)
    print(f"wrote {destination}")
