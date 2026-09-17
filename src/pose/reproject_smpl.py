"""Compare a SMPL pose with 2D keypoints in the calibrated broadcast image.

Usage: uv run python src/pose/reproject_smpl.py SNGS-043 1131

Requires data/models/yolo11n-pose.pt, downloaded by hand. The detector runs on
a padded player crop because a ~70 px broadcast player is too small full-frame.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
sys.path.insert(0, str(ROOT / "src" / "track"))
from compare_pnl import GOAL_SIDE, camera_matrix, pitch_to_soccernet, project_3d  # noqa: E402
from clips import frame_path  # noqa: E402

# COCO keypoint -> matching SMPL joint. Nose/eyes/ears have no exact SMPL match, so skip them.
COCO_TO_SMPL = np.array([[5, 16], [6, 17], [7, 18], [8, 19], [9, 20], [10, 21],
                         [11, 1], [12, 2], [13, 4], [14, 5], [15, 7], [16, 8]])
MIN_CONFIDENCE = 0.3


def iou(first, second):
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = np.prod(np.maximum(0, [x2 - x1, y2 - y1]))
    return intersection / (np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection)


def player_crop(image, box):
    """A 3x2 box around the player, returned with its image-space top-left corner."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left, top = max(0, int(x1 - width)), max(0, int(y1 - height / 2))
    right = min(image.shape[1], int(x2 + width))
    bottom = min(image.shape[0], int(y2 + height / 2))
    return image[top:bottom, left:right], np.array([left, top])


def observed_keypoints(model, image, tracked_box):
    crop, offset = player_crop(image, tracked_box)
    result = model(crop, device="mps", imgsz=640, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    boxes[:, [0, 2]] += offset[0]
    boxes[:, [1, 3]] += offset[1]
    chosen = max(range(len(boxes)), key=lambda i: iou(tracked_box, boxes[i]))
    points = result.keypoints.xy.cpu().numpy()[chosen] + offset
    confidence = result.keypoints.conf.cpu().numpy()[chosen]
    return points, confidence


def smpl_to_world(joints, pitch_xy, rotation, goal):
    """Anchor the average ankle at the tracked grass point, then undo camera rotation.

    HMR2's x/y/z joint axes match PnLCalib's camera axes here. This is checked by
    the final pixel metric; flipping y makes the median error about 82 px instead.
    """
    anchor = pitch_to_soccernet(np.asarray([pitch_xy]), goal)[0]
    foot = (joints[7] + joints[8]) / 2
    return anchor + (np.linalg.inv(rotation) @ (joints - foot).T).T


def main(clip, track_id, start, end, draw_frame):
    pose = np.load(ROOT / "data" / "pose" / "npz" / f"pose_{track_id}.npz")
    poses = dict(zip(pose["frame"], pose["joints"]))
    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    players = {f["frame"]: next((p for p in f["players"] if p["id"] == track_id), None)
               for f in tracks["frames"]}
    raw = json.loads((ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json").read_text())
    cameras = {f["frame"]: f["cam_params"] for f in raw["frames"] if f["ok"]}
    model_path = ROOT / "data/models/yolo11n-pose.pt"
    if not model_path.is_file():
        raise SystemExit(f"missing {model_path}; download it before this check")
    model, rows = YOLO(model_path), []
    for frame, joints in poses.items():
        if not start <= frame <= end or not players.get(frame) or frame not in cameras:
            continue
        player, camera = players[frame], cameras[frame]
        image = cv2.imread(str(frame_path(clip, frame)))
        observed = observed_keypoints(model, image, np.asarray(player["box_px"], float))
        if observed is None:
            continue
        keypoints, confidence = observed
        keep = confidence[COCO_TO_SMPL[:, 0]] > MIN_CONFIDENCE
        if keep.sum() < 6:
            continue
        projected = project_3d(camera_matrix(camera), smpl_to_world(
            joints, (player["x"], player["y"]), np.asarray(camera["rotation_matrix"]), GOAL_SIDE[clip]))
        errors = np.linalg.norm(projected[COCO_TO_SMPL[:, 1]] - keypoints[COCO_TO_SMPL[:, 0]], axis=1)
        rows.append((frame, projected, keypoints, keep, errors))
    if not rows:
        raise SystemExit("no frames with six confident keypoints")
    errors = np.concatenate([row[4][row[3]] for row in rows])
    print(f"{clip} track {track_id}: {len(rows)} frames, {len(errors)} confident joint matches")
    print(f"SMPL-joint reprojection error: median {np.median(errors):.1f}px, "
          f"mean {np.mean(errors):.1f}px, 95th percentile {np.percentile(errors, 95):.1f}px")
    chosen = next((row for row in rows if row[0] == draw_frame), rows[len(rows) // 2])
    image = cv2.imread(str(frame_path(clip, chosen[0])))
    for coco, smpl in COCO_TO_SMPL[chosen[3]]:
        cv2.circle(image, tuple(np.round(chosen[1][smpl]).astype(int)), 4, (0, 255, 0), -1)  # SMPL
        cv2.circle(image, tuple(np.round(chosen[2][coco]).astype(int)), 4, (0, 0, 255), -1)  # detector
    out = ROOT / "outputs" / f"smpl_reprojection_{clip}_{track_id}_{chosen[0]:06d}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), image)
    print(f"wrote {out.relative_to(ROOT)} (green = SMPL, red = 2D detector)")
    assert np.isclose(iou(np.array([0, 0, 2, 2]), np.array([1, 1, 3, 3])), 1 / 7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("track_id", type=int)
    parser.add_argument("--start", type=int, default=560)
    parser.add_argument("--end", type=int, default=620)
    parser.add_argument("--draw-frame", type=int, default=602)
    args = parser.parse_args()
    main(args.clip, args.track_id, args.start, args.end, args.draw_frame)
