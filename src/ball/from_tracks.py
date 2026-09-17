"""E1: the 2D ball track for a clip with no labels, from the detector's own ball boxes.

Usage: uv run python src/ball/from_tracks.py clip04

Writes data/ball/<clip>.json in exactly the shape from_labels.py writes, so A2's clean.py and
B's fit.py do not care which clip they are looking at. The difference is honesty about the
input: SoccerNet's rows are hand annotations, these are YOLO guesses. PLAN.md already records
why that matters on clip04 -- 2 to 4 "ball" boxes in 255 of 337 frames, because spare balls sit
by the ad boards -- so `ball_px` is only the most confident box, and some of those are wrong.

`pitch_px` is None here. The labels' version is a ray through their camera onto the grass; ours
would be a ray through OUR camera, which is the thing the fit is being checked against, so
computing it would just be quoting the fit back to itself.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "track"))
from clips import fps  # noqa: E402

DETECTOR = "football-player-detection-v9_botsort"


def read(clip):
    """The most confident "ball" box per frame. On clip04 this is NOT the ball, and it is measured.

    Consecutive pixel steps of 1690, 1682 and 1207 px (95% = 1127). Nothing on a pitch moves
    1690 px in 20 ms, so that is not the ball travelling -- it is the choice hopping between the
    match ball and the spare balls stacked by the ad boards. 199 of 296 frames offer more than
    one candidate, so the hop has plenty of chances.

    Tried and deleted (2026-09-18): pick the candidate NEAREST the last accepted one instead.
    It locked onto a spare ball -- 11 frames, pixel step median 1 px, max 2 px -- because the
    seed frame's most confident box is already a spare, and continuity then loyally follows it
    and rejects the real ball as too far. A continuity rule cannot rescue a detector that cannot
    tell the match ball from a ball on a rack; that is PLAN.md's D2 (`yolo-sn-ball-opt.pt`).
    """
    raw = json.loads((ROOT / "data" / "track" / f"{clip}_{DETECTOR}.json").read_text())
    rows = []
    for frame in raw["frames"]:
        balls = [b for b in frame["boxes"] if b["cls"] == "ball"]
        if not balls:
            continue
        best = max(balls, key=lambda b: b["conf"])
        x1, y1, x2, y2 = best["xyxy"]
        rows.append({"frame": frame["frame"], "px": [(x1 + x2) / 2, (y1 + y2) / 2],
                     "wh": [x2 - x1, y2 - y1], "pitch_px": None,
                     "conf": best["conf"], "candidates": len(balls)})
    return sorted(rows, key=lambda r: r["frame"])


def main(clip):
    rows = read(clip)
    frames = [r["frame"] for r in rows]
    missing = [f for f in range(frames[0], frames[-1] + 1) if f not in set(frames)]
    out = ROOT / "data" / "ball" / f"{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clip": clip, "fps": fps(clip), "missing": missing, "frames": rows}))
    contested = sum(r["candidates"] > 1 for r in rows)
    print(f"wrote {out}: ball in {len(rows)} of {frames[-1] - frames[0] + 1} frames, "
          f"{len(missing)} missing")
    print(f"  frames with more than one 'ball' box (so the pick could be wrong): {contested}")
    steps = sorted(((r["px"][0] - q["px"][0]) ** 2 + (r["px"][1] - q["px"][1]) ** 2) ** 0.5
                   for q, r in zip(rows, rows[1:]) if r["frame"] == q["frame"] + 1)
    if steps:
        print(f"  pixel step between frames: median {steps[len(steps) // 2]:.0f}, "
              f"95% {steps[int(len(steps) * 0.95)]:.0f}, max {steps[-1]:.0f}")
    widths = sorted(r["wh"][0] for r in rows)
    print(f"  box width px: min {widths[0]:.0f}, median {widths[len(widths) // 2]:.0f}, "
          f"max {widths[-1]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="clip04")
    main(parser.parse_args().clip)
