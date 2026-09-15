"""Stage 2: YOLO detection + tracking (BoT-SORT / ByteTrack) on a clip's frames. Saves raw boxes (pixels).

Usage: uv run python src/track/detect_track.py clip04 [model] [tracker]
  model = a file in data/models/. Default football-player-detection-v9.pt = Roboflow's football model
    ("player", "goalkeeper", "referee", "ball"); yolo26m.pt = general COCO model ("person", "sports ball").
    On clip04 the football model gives 32 person IDs vs 48, no photographers, ball in 274 vs 61 frames.
    Its class labels are unreliable on our match (the referee is called "player" 97% of the time).
  tracker = botsort.yaml (default) or bytetrack.yaml, both built into Ultralytics. Same people IDs on
    clip04 (32); BoT-SORT's camera-motion compensation keeps the ball in more frames (296 vs 274).
    Longer memory (track_buffer 150) and appearance re-ID changed nothing on clip04.
Reads data/frames/<clip>/*.jpg, writes data/track/<clip>_<model name>_<tracker name>.json:
  {"clip", "model", "frames": [{"frame": 0, "boxes": [{"id": 3, "cls": "player", "conf": 0.91,
                                                      "xyxy": [x1, y1, x2, y2]}]}]}
id = the tracker's track number (same player across frames, until an ID switch). Boxes the tracker
did not accept (too unsure) are left out.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
BALL = {"sports ball", "ball"}
KEEP = {"person", "player", "goalkeeper", "referee"} | BALL  # class names we want, from either model


def main(clip, model_name="football-player-detection-v9.pt", tracker="botsort.yaml"):
    model = YOLO(ROOT / "data" / "models" / model_name)  # yolo26m.pt downloads itself; the football model is downloaded by hand
    classes = [i for i, n in model.names.items() if n in KEEP]
    frames = []
    # stream=True: one frame at a time instead of all 337 results in memory.
    # persist=True: keep the tracker's memory between frames (that's what makes it tracking).
    # agnostic_nms=True: one box per person, even if the model is torn between "player" and "goalkeeper".
    # imgsz=1280: YOLO shrinks 1920 px to this; the default 640 makes far players ~10 px tall.
    for r in model.track(source=str(ROOT / "data" / "frames" / clip), tracker=tracker,
                         classes=classes, agnostic_nms=True, imgsz=1280, device="mps", stream=True, persist=True,
                         verbose=False):
        boxes = []
        if r.boxes.id is not None:
            for tid, c, conf, xyxy in zip(r.boxes.id.tolist(), r.boxes.cls.tolist(),
                                          r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                boxes.append({"id": int(tid), "cls": r.names[int(c)], "conf": round(conf, 3),
                              "xyxy": [round(v, 1) for v in xyxy]})
        frames.append({"frame": int(Path(r.path).stem), "boxes": boxes})

    out = ROOT / "data" / "track" / f"{clip}_{Path(model_name).stem}_{Path(tracker).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"clip": clip, "model": model_name, "tracker": tracker, "frames": frames}, open(out, "w"))

    people = [[b for b in f["boxes"] if b["cls"] not in BALL] for f in frames]
    ball_ids = {b["id"] for f in frames for b in f["boxes"] if b["cls"] in BALL}
    ids = {b["id"] for p in people for b in p} - ball_ids  # a ball track sometimes gets a person label
    balls = sum(any(b["cls"] in BALL for b in f["boxes"]) for f in frames)
    counts = {}
    for p in people:
        for b in p:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    print(f"{len(frames)} frames -> {out}")
    print(f"people per frame: min {min(map(len, people))}, max {max(map(len, people))}; unique IDs: {len(ids)}")
    print(f"boxes per class (all frames): {counts}")
    print(f"ball found in {balls}/{len(frames)} frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
