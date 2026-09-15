"""Draw tracked boxes + IDs on every frame and save a video, to check tracking by eye.

Usage: uv run python src/track/draw_tracks.py data/track/clip04_football-player-detection-v9_botsort.json
Writes outputs/<json name>.mp4. Each ID gets its own colour, so an ID switch shows up as a
player whose box suddenly changes colour and number.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 49.95  # clip04's frame rate (Stage 0)


def colour(tid):
    hue = (tid * 47) % 180  # spread neighbouring IDs far apart on the colour wheel
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def main(raw_path):
    raw_path = Path(raw_path)
    d = json.load(open(raw_path))
    out = ROOT / "outputs" / f"{raw_path.stem}.mp4"
    # Pipe raw frames into ffmpeg: H.264 + yuv420p so QuickTime and browsers can play it.
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", "1920x1080", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in d["frames"]:
        img = cv2.imread(str(ROOT / "data" / "frames" / d["clip"] / f"{f['frame']:05d}.jpg"))
        for b in f["boxes"]:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            c = colour(b["id"])
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, str(b["id"]), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(img, f"frame {f['frame']}", (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(d['frames'])} frames -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
