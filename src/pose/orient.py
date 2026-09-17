"""Turn HMR2's camera-space body into the viewer's pitch space.

HMR2 returns a body in the coordinates of the camera that saw it, so its facing only means
something relative to that camera: pan the camera and the same standing player "turns".
Stage 2 already fits the rotation from that camera to the pitch, so absolute facing is a
change of basis, not something to estimate.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from compare_pnl import GOAL_SIDE, pitch_to_soccernet  # noqa: E402

# SoccerNet world (X along the pitch, Y along the goal lines, Z DOWN) -> viewer scene
# (x = our pitch x, y = up, z = -our pitch y), read off pitch_to_soccernet: with our goal on the
# right X = 52.5 - y and Y = -x, so dx = -dY and dy = -dX; on the left X = -52.5 + y and Y = x.
# Both have determinant +1, so neither one mirrors the body.
WORLD_TO_SCENE = {"right": np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                  "left": np.array([[0.0, 1, 0], [0, 0, -1], [-1, 0, 0]])}


def intrinsics(cam):
    return np.array([[cam["x_focal_length"], 0, cam["principal_point"][0]],
                     [0, cam["y_focal_length"], cam["principal_point"][1]],
                     [0, 0, 1.0]])


def crop_to_full(K, centre_px):
    """HMR2 sees a crop, so its body sits in a camera aimed at the crop centre rather than down
    the optical axis. Rotate the optical axis onto the ray through that centre (the CLIFF
    correction). Worth a few degrees for a player away from the middle of the frame."""
    ray = np.linalg.inv(K) @ np.array([centre_px[0], centre_px[1], 1.0])
    ray /= np.linalg.norm(ray)
    axis = np.cross([0, 0, 1.0], ray)
    sine = np.linalg.norm(axis)
    if sine < 1e-9:
        return np.eye(3)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + skew + skew @ skew * ((1 - ray[2]) / sine ** 2)


def scene_rotations(clip, track_id, frames):
    """(N, 3, 3): HMR2 camera axes -> viewer scene axes, one rotation per frame."""
    raw = json.loads((ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json").read_text())
    camera = json.loads((ROOT / "data" / "camera" / f"{clip}.json").read_text())
    params = {f["frame"]: f["cam_params"] for f in raw["frames"] if f["ok"]}
    # export_camera.py already decided which frames are broken or jumpy, and those cameras are
    # wrong by metres. Borrow the nearest frame it kept instead; on a pan that costs a little yaw.
    trusted = sorted({f["frame"] for f in camera["frames"] if not f["filled"]} & set(params))
    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    boxes = {f["frame"]: p["box_px"] for f in tracks["frames"] for p in f["players"]
             if p["id"] == track_id and p.get("box_px")}
    world_to_scene = WORLD_TO_SCENE[GOAL_SIDE[clip]]
    rotations = []
    for frame in frames:
        cam = params[min(trusted, key=lambda good: abs(good - int(frame)))]
        K = intrinsics(cam)
        box = boxes.get(int(frame))
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else (K[0, 2], K[1, 2])
        camera_to_world = np.array(cam["rotation_matrix"]).T
        rotations.append(world_to_scene @ camera_to_world @ crop_to_full(K, centre))
    return np.array(rotations)


def scene_yaw(rotations, global_orient):
    """Body facing in the viewer's ground plane, radians, unwrapped.

    SMPL's canonical body looks down +z, so the forward direction is the third column of the
    rotation chain. Yaw is measured the same way the viewer's minimap is: atan2(x, z).
    """
    forward = (rotations @ global_orient)[:, :, 2]
    return np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))


def yaw_smoothing(yaw, window):
    """Rotations about the scene's up axis that replace each frame's yaw with a moving average.

    Only yaw is touched: the body's lean stays HMR2's. Smoothing belongs here and not in camera
    space, where a moving average also averages in the camera's own pan.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be an odd positive number")
    pad = window // 2
    smooth = np.convolve(np.pad(yaw, pad, mode="edge"), np.ones(window) / window, mode="valid")
    cos, sin = np.cos(smooth - yaw), np.sin(smooth - yaw)
    out = np.zeros((len(yaw), 3, 3))
    out[:, 0, 0], out[:, 0, 2] = cos, sin
    out[:, 1, 1] = 1
    out[:, 2, 0], out[:, 2, 2] = -sin, cos
    return out


def demo():
    """WORLD_TO_SCENE has to agree with the pitch_to_soccernet the rest of the pipeline uses."""
    for goal in ("left", "right"):
        step = pitch_to_soccernet(np.array([[1.0, 2.0]]), goal) - pitch_to_soccernet(np.zeros((1, 2)), goal)
        scene = WORLD_TO_SCENE[goal] @ step[0]
        assert np.allclose(scene, [1.0, 0.0, -2.0]), (goal, scene)  # x stays x, y becomes -z
        assert np.isclose(np.linalg.det(WORLD_TO_SCENE[goal]), 1.0), goal  # a rotation, not a mirror
    # Straight down the optical axis the crop correction does nothing; off to one side it turns.
    K = intrinsics({"x_focal_length": 1500, "y_focal_length": 1500, "principal_point": [960, 540]})
    assert np.allclose(crop_to_full(K, (960, 540)), np.eye(3))
    turned = crop_to_full(K, (1460, 540)) @ [0, 0, 1.0]
    assert np.isclose(np.degrees(np.arctan2(turned[0], turned[2])), np.degrees(np.arctan2(500, 1500)))
    # Smoothing a constant yaw must change nothing, and the window must be odd.
    assert np.allclose(yaw_smoothing(np.full(7, 1.3), 5), np.eye(3))
    # A body facing world +X, with our goal on the right, faces the scene's +z: away from that
    # goal, because pitch_to_soccernet puts X = 52.5 - y there. Turn it a quarter and yaw moves 90.
    towards_far_goal = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # canonical +z -> world +X
    yaw = scene_yaw(WORLD_TO_SCENE["right"][None], towards_far_goal[None])
    assert np.isclose(np.degrees(yaw[0]), 0.0), np.degrees(yaw)
    print("orient self-check ok")


if __name__ == "__main__":
    demo()
