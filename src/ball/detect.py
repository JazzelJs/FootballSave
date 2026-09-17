"""D2: how often does SoccerNet-v3D's ball detector actually find the ball?

Usage: uv run python src/ball/detect.py SNGS-043
       uv run python src/ball/detect.py SNGS-043 --imgsz 1280

Measured on SNGS-043 on purpose: its labels say exactly which 738 of 750 frames contain a ball,
so this is a real detection rate and not a guess. The paper warns these models transfer poorly
to unfamiliar camera setups, so the number matters more than the model's reputation.

The thing to watch is `imgsz`. The ball is 3-25 px wide (median 12) in a 1920x1080 frame. YOLO
letterboxes the frame down to imgsz first, so at the default 640 a 12 px ball arrives as **4 px**
-- smaller than one stride-8 output cell. Running at a bigger imgsz costs time and buys pixels,
and the point of this script is to find out how much.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "track"))
from clips import frame_path  # noqa: E402

MODEL = ROOT / "data" / "models" / "yolo-sn-ball-opt.pt"
CONF = 0.10   # deliberately low: give the model its best chance, then look at what it found
HIT_PX = 25.0  # centre-to-centre distance that counts as "found the ball" (the ball is ~12 px)


def truth(clip):
    """frame -> the label's ball pixel, for the frames the annotators say contain a ball."""
    ball = json.loads((ROOT / "data" / "ball" / f"{clip}.json").read_text())
    return {r["frame"]: np.array(r["px"]) for r in ball["frames"]}


def run(clip, imgsz):
    labels = truth(clip)
    frames = sorted(labels) + [f for f in range(min(labels), max(labels) + 1) if f not in labels]
    model = YOLO(MODEL)
    hits, errors, missed, wrong_object, false_alarms = 0, [], 0, 0, 0
    started = time.time()
    for frame in sorted(frames):
        path = frame_path(clip, frame)
        result = model.predict(str(path), imgsz=imgsz, conf=CONF, verbose=False, device="mps")[0]
        boxes = result.boxes.xywh.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        if frame not in labels:
            false_alarms += len(boxes) > 0   # no ball in this frame, so anything found is wrong
            continue
        if len(boxes) == 0:
            missed += 1
            continue
        # The nearest box, not the most confident: we are measuring whether the ball was FOUND,
        # and separately how often the most confident thing is some other ball entirely.
        distances = np.linalg.norm(boxes[:, :2] - labels[frame], axis=1)
        nearest = distances.min()
        if nearest <= HIT_PX:
            hits += 1
            errors.append(nearest)
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            wrong_object += distances[best] > HIT_PX
        else:
            missed += 1
    seconds = time.time() - started
    labelled = len(labels)
    errors = np.array(errors) if errors else np.array([np.nan])
    print(f"imgsz {imgsz}: ball found in {hits} of {labelled} labelled frames "
          f"= {100 * hits / labelled:.1f}%")
    print(f"  missed {missed}; centre error median {np.nanmedian(errors):.1f} px, "
          f"95% {np.nanpercentile(errors, 95):.1f} px")
    print(f"  most confident box was a DIFFERENT ball in {wrong_object} of the {hits} hits")
    print(f"  false alarms on the {len(frames) - labelled} ball-free frames: {false_alarms}")
    print(f"  {seconds:.0f} s for {len(frames)} frames ({seconds / len(frames) * 1000:.0f} ms each)")
    return {"imgsz": imgsz, "rate": hits / labelled, "median_px": float(np.nanmedian(errors)),
            "wrong_object": int(wrong_object), "false_alarms": int(false_alarms), "seconds": seconds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="SNGS-043")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920])
    args = parser.parse_args()
    out = [run(args.clip, size) for size in args.imgsz]
    (ROOT / "data" / "ball" / f"{args.clip}_detect.json").write_text(json.dumps(out))
