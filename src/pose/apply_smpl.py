"""Apply SMPL pose parameters and export joints for the football viewer.

Input NPZ fields are optional and use SMPL conventions:
    body_pose:    (frames, 69)
    global_orient:(frames, 3)
    body_pose_rotmat:    (frames, 23, 3, 3) from HMR2
    global_orient_rotmat:(frames, 1, 3, 3) from HMR2
    betas:        (10,) or (frames, 10)
    transl:       (frames, 3)

With no input, this writes one neutral-pose frame as a pipeline smoke test.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from smpl import load_model


def frames(value, count, width):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = np.repeat(array[None], count, axis=0)
    if array.shape != (count, width):
        raise ValueError(f"expected ({count}, {width}), got {array.shape}")
    return array


def export(input_path: Path | None, output_path: Path):
    source = np.load(input_path) if input_path else {}
    if input_path and "body_pose" not in source and "body_pose_rotmat" not in source:
        raise ValueError(
            "input has no SMPL pose parameters; rerun the updated Kaggle inference cell"
        )
    count = len(source["body_pose_rotmat"]) if "body_pose_rotmat" in source else len(source["body_pose"]) if "body_pose" in source else 1

    transl = frames(source.get("transl", np.zeros(3)), count, 3)
    # Body shape = 0 = the average adult male, on purpose. HMR2 predicts betas for SMPL's NEUTRAL
    # body, and the same 10 numbers mean something else in the MALE model: decoded here they gave
    # track 166 a female-looking chest and hips. Joint rotations DO carry over between the two
    # models (same 24-joint tree), so the motion stays HMR2's and only the build is swapped.
    betas = np.zeros(10, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.repeat(betas[None], count, axis=0)
    if betas.shape != (count, 10):
        raise ValueError(f"expected ({count}, 10), got {betas.shape}")

    model = load_model()
    hmr2_rotations = "body_pose_rotmat" in source
    if hmr2_rotations:
        body_pose = np.asarray(source["body_pose_rotmat"], dtype=np.float32)
        global_orient = np.asarray(source["global_orient_rotmat"], dtype=np.float32)
        if body_pose.shape != (count, 23, 3, 3):
            raise ValueError(f"expected ({count}, 23, 3, 3), got {body_pose.shape}")
        if global_orient.shape != (count, 1, 3, 3):
            raise ValueError(f"expected ({count}, 1, 3, 3), got {global_orient.shape}")
    else:
        body_pose = frames(source.get("body_pose", np.zeros(69)), count, 69)
        global_orient = frames(source.get("global_orient", np.zeros(3)), count, 3)

    with torch.no_grad():
        result = model(
            body_pose=torch.from_numpy(body_pose),
            global_orient=torch.from_numpy(global_orient),
            transl=torch.from_numpy(transl),
            betas=torch.from_numpy(betas),
            pose2rot=not hmr2_rotations,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        joints=result.joints[:, :24].cpu().numpy(),
        vertices=result.vertices.cpu().numpy(),
        frame=np.asarray(source["frame"]) if "frame" in source and len(source["frame"]) == count else np.arange(count),
        fps=np.float32(source["fps"]) if "fps" in source else np.float32(25),
    )
    print(f"wrote {output_path} ({count} frame{'s' if count != 1 else ''})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="NPZ containing SMPL pose parameters")
    parser.add_argument("--output", type=Path, default=Path("outputs/pose_smoke_test.npz"))
    args = parser.parse_args()
    export(args.input, args.output)
