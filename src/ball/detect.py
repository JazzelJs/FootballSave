"""D2 + E1: SoccerNet-v3D's ball detector -- how often it finds the ball, and the track it gives.

Usage: uv run python src/ball/detect.py SNGS-043              # measure the rate (D2)
       uv run python src/ball/detect.py clip04 --write        # write data/ball/clip04.json (E1)

Replaces the deleted from_tracks.py, which read "ball" boxes out of the player detector and got
a track that was not the ball: pixel steps of 1690 px, because the most confident box hops
between the match ball and the spare balls by the ad boards.

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
IMGSZ = 1920   # full resolution. Measured on SNGS-043: 3.9% at 640, 43.9% at 1280, 53.1% at 1920.
MARGIN_M = 3.0   # how far outside the touchline/goal line a ball may still be the match ball
STATIC_M = 1.0   # metres: detections landing in the same square metre are the same object
STATIC_SHARE = 0.25  # an object sitting still in more than this share of frames is furniture


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


def write_track(clip, imgsz=IMGSZ):
    """E1: the most confident ball box per frame -> data/ball/<clip>.json, the A1 format.

    Deliberately the same shape from_labels.py writes, so clean.py and fit.py do not care which
    clip they are looking at. `pitch_px` is None: the labels' version is a ray through THEIR
    camera, and ours would be a ray through the camera the fit is being checked against.

    The honest caveat, measured on SNGS-043 where we can check: the most confident box is a
    different ball in 30 of 392 hits. On a clip with spare balls by the ad boards that number
    will be worse, so look at the printed step distribution before trusting the track.
    """
    sys.path.insert(0, str(ROOT / "src" / "track"))
    sys.path.insert(0, str(ROOT / "src" / "calib"))
    from clips import fps  # noqa: PLC0415  local: only this path needs it
    from homography import project  # noqa: PLC0415
    camera = json.loads((ROOT / "data" / "camera" / f"{clip}.json").read_text())
    inverse = {f["frame"]: np.linalg.inv(np.array(f["H"])) for f in camera["frames"]}
    model = YOLO(MODEL)
    frames = sorted(int(p.stem) for p in frame_path(clip, 0).parent.glob("*.jpg"))

    # Pass 1: every candidate, so the filters below can look at the clip as a whole.
    found = {}
    for frame in frames:
        result = model.predict(str(frame_path(clip, frame)), imgsz=imgsz, conf=CONF,
                               verbose=False, device="mps")[0]
        if len(result.boxes):
            found[frame] = (result.boxes.xywh.cpu().numpy(), result.boxes.conf.cpu().numpy())

    # Which detections never move? On clip04 the detector keeps finding a ball-shaped object at
    # grass (20.0, -2.1 m), just behind the goal line, and at frame 182 it even outscores the real
    # ball, 0.39 to 0.38. A spare ball on a rack is still; the match ball is not.
    #
    # The clustering MUST happen on the grass, not in the image. A first version binned pixels and
    # found zero static objects, because the camera pans: a world-static object slides across the
    # frame, so in pixels nothing is ever still. Through inv(H) it sits in the same square metre
    # all clip. (The on-pitch test alone does not catch it either -- y = -2.1 m is inside the 3 m
    # margin that the real ball needs, and it moved the track by one frame out of 316.)
    cells = {}
    for frame, (boxes, _) in found.items():
        if frame not in inverse:
            continue
        for grass in project(inverse[frame], boxes[:, :2]):
            cells.setdefault((round(grass[0] / STATIC_M), round(grass[1] / STATIC_M)), set()).add(frame)
    furniture = {cell for cell, seen in cells.items() if len(seen) > STATIC_SHARE * len(frames)}

    rows, dropped_static, dropped_offpitch = [], 0, 0
    for frame in frames:
        if frame not in found:
            continue
        boxes, confidences = found[frame]
        grass_all = project(inverse[frame], boxes[:, :2]) if frame in inverse else None
        keep = (np.array([(round(g[0] / STATIC_M), round(g[1] / STATIC_M)) not in furniture
                          for g in grass_all]) if grass_all is not None
                else np.ones(len(boxes), dtype=bool))
        dropped_static += int((~keep).sum())
        if not keep.any():
            continue
        boxes, confidences = boxes[keep], confidences[keep]
        if frame in inverse:  # and anything well off the pitch is not the ball either
            grass = project(inverse[frame], boxes[:, :2])
            on_pitch = ((np.abs(grass[:, 0]) < 34.0 + MARGIN_M)
                        & (grass[:, 1] > -MARGIN_M) & (grass[:, 1] < 105.0 + MARGIN_M))
            dropped_offpitch += int((~on_pitch).sum())
            if not on_pitch.any():
                continue
            boxes, confidences = boxes[on_pitch], confidences[on_pitch]
        best = int(confidences.argmax())
        x, y, w, h = boxes[best]
        rows.append({"frame": frame, "px": [float(x), float(y)], "wh": [float(w), float(h)],
                     "pitch_px": None, "conf": float(confidences[best])})

    missing = [f for f in range(frames[0], frames[-1] + 1) if f not in {r["frame"] for r in rows}]
    out = ROOT / "data" / "ball" / f"{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clip": clip, "fps": fps(clip), "missing": missing, "frames": rows}))
    steps = sorted(float(np.linalg.norm(np.array(b["px"]) - np.array(a["px"])))
                   for a, b in zip(rows, rows[1:]) if b["frame"] == a["frame"] + 1)
    print(f"wrote {out}: ball in {len(rows)} of {len(frames)} frames, {len(missing)} missing")
    print(f"  dropped {dropped_static} detections on {len(furniture)} object(s) that never move, "
          f"{dropped_offpitch} off the pitch")
    print(f"  pixel step: median {steps[len(steps) // 2]:.0f}, "
          f"95% {steps[int(len(steps) * 0.95)]:.0f}, max {steps[-1]:.0f}")
    pairs = [(b["frame"], float(np.linalg.norm(np.array(b["px"]) - np.array(a["px"]))))
             for a, b in zip(rows, rows[1:]) if b["frame"] == a["frame"] + 1]
    biggest = sorted(pairs, key=lambda fs: -fs[1])[:6]
    print("  biggest steps (a kick, or the pick swapping to another ball): "
          + ", ".join(f"{frame} ({step:.0f} px)" for frame, step in biggest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="SNGS-043")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920])
    parser.add_argument("--write", action="store_true", help="E1: write the ball track, don't measure")
    args = parser.parse_args()
    if args.write:
        write_track(args.clip)
    else:
        out = [run(args.clip, size) for size in args.imgsz]
        (ROOT / "data" / "ball" / f"{args.clip}_detect.json").write_text(json.dumps(out))
