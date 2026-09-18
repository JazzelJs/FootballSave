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


def pose_npz(clip, track_id):
    """Where a track's HMR2 output lives. The CLIP is part of the name, not decoration.

    A track id means nothing without its clip: SNGS-043 has a track 17 and so does clip04, and
    the flat `pose_17.npz` they both wanted cost us a near-miss -- the browser saved clip04's
    download as `pose_17-3.npz` rather than overwriting SNGS-043's, which is the only reason the
    older file survived. Three scripts built that flat path, so the collision was in the code too,
    not just in the download folder.
    """
    return ROOT / "data" / "pose" / "npz" / f"pose_{clip}_{track_id}.npz"


def pose_mesh(clip, track_id):
    """The viewer's mesh for a track, named the same way and for the same reason."""
    return ROOT / "data" / "pose" / "mesh" / f"pose_{clip}_{track_id}.smpl"


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
    """Odd moving-average window over the pitch-space yaw, returned as rotations.

    HOW TO PICK `window`, because the obvious method does not work. Sweeping it against the
    direction-of-travel proxy (`orientation_error.py`) gives a median that falls MONOTONICALLY and
    never turns around -- on clip04 track 17: 16.7° at w=1, 13.9° at 15, 11.3° at 31, 11.6° at 41
    (0.82 s). "Optimise the window" therefore means "smooth until the signal is gone", and a
    0.8 s moving average erases real turning: a player can turn 180° in half a second.

    The proxy prefers smoothing because the proxy is ITSELF smooth -- travel direction comes from
    tracker positions 4 frames apart. So it rewards any change that makes the pose yaw smoother,
    whether or not the pose is more correct. Never tune a parameter against a metric it can game.

    What the sweep DOES say: the whole range w=1..15 moves the median by 0.8-3.3° on every track
    of both clips, which is inside the proxy's own noise. The window barely matters, so pick it
    on physics and keep it consistent: match it in TIME across clips. Raw per-frame yaw change is
    1.5-2.2°/frame on clip04 against 3.6-4.2° on SNGS-043 -- a ratio of about 2, exactly the frame
    rate ratio (49.95 / 25), so it is the same angular noise spread over twice as many frames.
    SNGS-043 uses w=5 at 25 fps = 0.20 s, so clip04 uses w=9 at 49.95 fps = 0.18 s.
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
