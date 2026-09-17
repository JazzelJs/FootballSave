"""Draw each posed player's facing direction on the broadcast frame, to look at it.

Usage: uv run python src/pose/draw_facing.py SNGS-043 602 284 17 171 1131

An arrow 2 m long lies on the grass at the player's feet, pointing where the SMPL body faces
after orient.py rotates it into pitch space. If the chain is right the arrow agrees with the
player in the picture; a mirrored or camera-space body points somewhere else entirely.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import orient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
sys.path.insert(0, str(ROOT / "src" / "track"))
from compare_pnl import GOAL_SIDE, camera_matrix, pitch_to_soccernet, project_3d  # noqa: E402
from clips import frame_path  # noqa: E402

ARROW_M = 2.0
STEP = 2  # frames either side, the same span orientation_error.py measures speed over


def facing_on_grass(clip, track_id, frame):
    """(pitch dx, dy) unit vector: where this track's body faces on the grass at this frame."""
    source = np.load(ROOT / "data" / "pose" / "npz" / f"pose_{track_id}.npz")
    frames = np.asarray(source["frame"], dtype=int)
    index = np.flatnonzero(frames == frame)
    if len(index) != 1:
        return None
    global_orient = np.asarray(source["global_orient_rotmat"], dtype=float)[:, 0]
    yaw = orient.scene_yaw(orient.scene_rotations(clip, track_id, frames), global_orient)[index[0]]
    # scene z = -pitch y, so the arrow's pitch components come back with that sign undone.
    return np.array([np.sin(yaw), -np.cos(yaw)])


def main(clip, frame, track_ids):
    raw = json.loads((ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json").read_text())
    cam = next(f["cam_params"] for f in raw["frames"] if f["frame"] == frame and f["ok"])
    P = camera_matrix(cam)
    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    players = {p["id"]: p for f in tracks["frames"] if f["frame"] == frame for p in f["players"]}
    around = {f["frame"]: {p["id"]: p for p in f["players"]}
              for f in tracks["frames"] if abs(f["frame"] - frame) == STEP}
    image = cv2.imread(str(frame_path(clip, frame)))
    drawn = 0
    for track_id in track_ids:
        player, facing = players.get(track_id), facing_on_grass(clip, track_id, frame)
        if player is None or facing is None:
            print(f"track {track_id}: no box or no pose at frame {frame}")
            continue
        start = np.array([player["x"], player["y"]])
        ends = pitch_to_soccernet(np.array([start, start + ARROW_M * facing]), GOAL_SIDE[clip])
        (x1, y1), (x2, y2) = project_3d(P, ends).astype(int)
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 220, 255), 3, tipLength=0.3)
        # Red = where the tracker says the player is travelling. A running player faces where they
        # run, so the two arrows should agree; where they do not, one of them is worth doubting.
        before = around.get(frame - STEP, {}).get(track_id)
        after = around.get(frame + STEP, {}).get(track_id)
        if before and after:
            step = np.array([after["x"] - before["x"], after["y"] - before["y"]])
            speed = np.linalg.norm(step) / (2 * STEP / tracks["fps"])
            if speed >= 1.0:
                travel = pitch_to_soccernet(
                    np.array([start, start + ARROW_M * step / np.linalg.norm(step)]), GOAL_SIDE[clip])
                (a1, b1), (a2, b2) = project_3d(P, travel).astype(int)
                cv2.arrowedLine(image, (a1, b1), (a2, b2), (0, 0, 255), 2, tipLength=0.3)
        if player.get("box_px"):
            box = np.asarray(player["box_px"], dtype=int)
            cv2.rectangle(image, box[:2], box[2:], (0, 220, 0), 2)
            cv2.putText(image, str(track_id), (box[0], box[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        drawn += 1
    out = ROOT / "outputs" / f"facing_{clip}_{frame:06d}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), image)
    print(f"wrote {out} ({drawn} player{'s' if drawn != 1 else ''})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("frame", type=int)
    parser.add_argument("track_ids", type=int, nargs="+")
    args = parser.parse_args()
    main(args.clip, args.frame, args.track_ids)
