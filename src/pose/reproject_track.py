"""Check a tracked player's pitch position against the calibrated image.

Usage: uv run python src/pose/reproject_track.py SNGS-043 1131 --start 560 --end 620
Writes an annotated kick-frame image and prints the pixel-error summary.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def project(H, xy):
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def main(clip, track_id, start, end, draw_frame):
    tracks = json.loads((ROOT / "data" / "tracks" / f"{clip}.json").read_text())
    cameras = json.loads((ROOT / "data" / "camera" / f"{clip}.json").read_text())
    H_by_frame = {f["frame"]: np.asarray(f["H"], dtype=float) for f in cameras["frames"]}
    rows = []
    for frame in tracks["frames"]:
        n = frame["frame"]
        if not start <= n <= end or n not in H_by_frame:
            continue
        player = next((p for p in frame["players"] if p["id"] == track_id), None)
        if not player or not player.get("box_px"):
            continue
        box = player["box_px"]
        observed = np.array([(box[0] + box[2]) / 2, box[3]])
        expected = project(H_by_frame[n], (player["x"], player["y"]))
        rows.append({"frame": n, "expected": expected, "observed": observed,
                     "error_px": float(np.linalg.norm(expected - observed))})
    if not rows:
        raise SystemExit("no tracked frames in requested range")
    errors = np.array([r["error_px"] for r in rows])
    print(f"{clip} track {track_id}: {len(rows)} frames, "
          f"median {np.median(errors):.1f}px, mean {np.mean(errors):.1f}px, "
          f"max {np.max(errors):.1f}px")
    print("largest errors:", ", ".join(f"{r['frame']}={r['error_px']:.1f}px"
          for r in sorted(rows, key=lambda r: r["error_px"], reverse=True)[:5]))

    chosen = next((r for r in rows if r["frame"] == draw_frame), rows[len(rows) // 2])
    image_path = ROOT / "data" / "soccernet" / clip / "img1" / f"{chosen['frame']:06d}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"missing frame image: {image_path}")
    frame = next(f for f in tracks["frames"] if f["frame"] == chosen["frame"])
    player = next(p for p in frame["players"] if p["id"] == track_id)
    x1, y1, x2, y2 = map(int, map(round, player["box_px"]))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    u, v = map(int, np.round(chosen["expected"]))
    cv2.drawMarker(image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(image, f"track {track_id}  reprojection error {chosen['error_px']:.1f}px",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    out = ROOT / "outputs" / f"reprojection_{clip}_{chosen['frame']:06d}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), image)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("track_id", type=int)
    parser.add_argument("--start", type=int, default=560)
    parser.add_argument("--end", type=int, default=620)
    parser.add_argument("--draw-frame", type=int, default=602)
    args = parser.parse_args()
    main(args.clip, args.track_id, args.start, args.end, args.draw_frame)
