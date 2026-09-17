"""The ball's 2D track, read out of a SoccerNet clip's labels.

Usage: uv run python src/ball/from_labels.py SNGS-043

Writes data/ball/<clip>.json: one row per labelled frame. Two things live in a row and they are
NOT the same thing:
  px          the ball in the picture. This is a real measurement.
  pitch_px    where a ray through that pixel hits the GRASS, as the labels compute it. While the
              ball is in the air this is its shadow, not the ball -- at frame 620 of SNGS-043 it
              sits at X = 53.5 m, a metre behind a goal line that is at 52.5 m.
Only `px` is input to a trajectory fit. `pitch_px` is for comparing against, and for the frames
before the kick where the ball really is on the grass.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "track"))
from clips import fps  # noqa: E402

BALL = 4  # category_id in Labels-GameState.json


def read(clip):
    labels = json.loads((ROOT / "data" / "soccernet" / clip / "Labels-GameState.json").read_text())
    frame_of = {image["image_id"]: int(image["file_name"].split(".")[0]) for image in labels["images"]}
    rows = []
    for a in labels["annotations"]:
        if a["category_id"] != BALL:
            continue
        box, pitch = a["bbox_image"], a.get("bbox_pitch")
        rows.append({"frame": frame_of[a["image_id"]],
                     "px": [box["x_center"], box["y_center"]],
                     "wh": [box["w"], box["h"]],
                     "pitch_px": None if pitch is None else
                                 [pitch["x_bottom_middle"], pitch["y_bottom_middle"]]})
    return sorted(rows, key=lambda r: r["frame"])


def main(clip):
    rows = read(clip)
    frames = [r["frame"] for r in rows]
    missing = [f for f in range(frames[0], frames[-1] + 1) if f not in set(frames)]
    out = ROOT / "data" / "ball" / f"{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clip": clip, "fps": fps(clip), "missing": missing, "frames": rows}))
    widths = sorted(r["wh"][0] for r in rows)
    print(f"wrote {out}: ball in {len(rows)} of {frames[-1] - frames[0] + 1} frames "
          f"({frames[0]}–{frames[-1]}), {len(missing)} missing")
    print(f"  gaps: {missing}")
    print(f"  box width px: min {widths[0]}, median {widths[len(widths) // 2]}, max {widths[-1]}")


def demo():
    """The labels must survive the round trip, and the shadow must misbehave where it should."""
    rows = {r["frame"]: r for r in read("SNGS-043")}
    assert len(rows) == 738, len(rows)
    assert rows[602]["px"] == [941.0, 560.0], rows[602]
    # Before the kick the ball is on the grass, so its shadow is inside the pitch...
    assert rows[601]["pitch_px"][0] < 52.5, rows[601]
    # ...and eighteen frames later the same computation puts it behind the goal line. That is the
    # ray problem, and the reason this stage exists.
    assert rows[620]["pitch_px"][0] > 52.5, rows[620]
    print("from_labels self-check ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="SNGS-043")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    demo() if args.self_check else main(args.clip)
