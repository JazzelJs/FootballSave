"""A2: clean the label ball track -- fill the gaps, throw out what isn't the ball.

Usage: uv run python src/ball/clean.py SNGS-043
       uv run python src/ball/clean.py --self-check

Reads  data/ball/<clip>.json  (written by from_labels.py)
Writes data/ball/<clip>_clean.json, same row shape plus one new key:
    source   "label"   straight from the annotations, a real measurement
             "filled"  invented by us to close a gap
             "dropped" a labelled row we decided not to trust (kept, so it stays visible)
Only "label" rows may feed the fit in B. That is the whole point of the extra key: a fit that
quietly eats our own inventions cannot be checked against anything.

What the data actually looks like (measured 2026-09-18, don't re-derive it):

  The three gaps are ALL before the kick, none inside the flight window 602-622.
    367            1 frame.  366 = (1218.5, 637.5) -> 368 = (1234.5, 643.5), moving steadily.
    476-481        6 frames. 475 = (984, 628)      -> 482 = (1069, 592). 92 px over 7 frames.
    558-562        5 frames. 557 = (1300.5, 602.5) -> 563 = (1156.5, 614.5).
                   Look at that last one before you write a loop: over 555-557 the ball is going
                   RIGHT (1240 -> 1300) and it comes back 144 px to the LEFT. A straight line
                   between the two ends does not describe whatever happened in between.

  The small boxes: 14 rows are <= 4 px wide, at frames 29-34, 62-68, 100, 101, 134. All in the
  first 5 seconds, none near the shot. A 3 px ball is 4 label pixels of evidence; what else could
  sit there, and does it matter for this clip?

  41 rows repeat the previous frame's pixel EXACTLY. Three of them are inside the flight:
    615 -> 616  step 3.0 px      then 616 -> 617  29.0 px
    617 -> 618  step 4.2 px      then 618 -> 619  27.1 px
    620 -> 621  step 3.6 px      then 621 -> 622   9.3 px
  Everywhere else in the flight the step is 15-25 px/frame. A ball at ~100 km/h does not pause
  for 40 ms. Each small step is followed by a double-sized one: the pair together covers the
  normal distance. So what is frame 616, and is it a measurement you want in a least-squares sum?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "track"))

FLIGHT = (602, 622)  # the fit window from the plan, only used by the self-check below
DIR_AGREE = 0.0    # cos of the angle a gap may bend: 0 = fill only while the ball still turns < 90 deg
RATIO = 0.35       # a step below this fraction of the recent median step is a mistimed label
MIN_MOVE = 1.0     # px: below this the ball is standing still and the ratio means nothing
WINDOW = 5         # how many steps BEFORE a frame set its yardstick (see suspicious())


def fill_gaps(rows, missing):
    """Give every frame in `missing` a row -- but only where the ball kept going the same way.

    rows:    list of A1 rows, sorted by frame, each {"frame", "px", "wh", "pitch_px"}
    missing: the frame numbers with no label (from the A1 json)
    Returns: the same list with new rows added, still sorted, the new ones "source": "filled".

    The rule is one line of geometry: compare the direction the ball was travelling just before
    the gap with the direction of the straight line across it. If they agree, the ball carried on
    and a straight line is a fair guess. If they disagree, the ball did something we did not see
    and a straight line is not a guess, it is an invention.
        gap 367        before (0.94, 0.34)   chord (0.94, 0.35)   dot  1.00  -> fill
        gap 476-481    before (-0.76, 0.65)  chord (0.92, -0.39)  dot -0.95  -> refuse
        gap 558-562    before (1.00, 0.05)   chord (-1.00, 0.08)  dot -0.99  -> refuse
    Both refused gaps are a pass or a clearance: the ball reverses across them. Neither is inside
    the flight window, so refusing costs the fit in B nothing at all.
    """
    by_frame = {r["frame"]: r for r in rows}
    filled = []
    for gap in _runs(missing):
        start, end = gap[0] - 1, gap[-1] + 1
        before, after = by_frame.get(start), by_frame.get(end)
        if before is None or after is None or (start - 1) not in by_frame:
            continue  # no direction to check against, so no licence to invent
        if _unit(by_frame[start - 1], before) @ _unit(before, after) < DIR_AGREE:
            continue
        for frame in gap:
            t = (frame - start) / (end - start)
            filled.append({"frame": frame,
                           "px": [_lerp(before["px"][i], after["px"][i], t) for i in (0, 1)],
                           "wh": [_lerp(before["wh"][i], after["wh"][i], t) for i in (0, 1)],
                           "pitch_px": None,  # a guessed pixel has no business claiming a shadow
                           "source": "filled"})
    return sorted(rows + filled, key=lambda r: r["frame"])


def _runs(frames):
    """[367, 476, 477, 478] -> [[367], [476, 477, 478]]: consecutive frame numbers grouped."""
    runs = []
    for frame in sorted(frames):
        if runs and runs[-1][-1] == frame - 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def _unit(a, b):
    """Unit vector in pixels from row a to row b. Zero-length stays zero (it agrees with nothing)."""
    v = np.array(b["px"], dtype=float) - np.array(a["px"], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _lerp(a, b, t):
    return a + (b - a) * t


def suspicious(rows):
    """Drop rows whose annotation barely moved while the ball was travelling: they are mistimed.

    The defect, in the flight: three steps of 3.0 / 4.2 / 3.6 px where the ball is otherwise
    covering 15-25 px per frame, and each one is followed by a double-length step. The pair
    together travels the normal distance, so the ball never paused -- the annotation lagged a
    frame, then caught up. A point pinned to the wrong instant is worse in a least-squares sum
    than no point at all, because the fit has no way to know it is late.

    The test is a ratio, not a pixel count, so there is nothing to re-pick per clip: compare each
    step with the median of the five steps BEFORE it. Past-only matters -- a window that also
    looks forward pulls frame 621's yardstick down with the post-622 frames where the ball is
    already in the net, and 621 survives a defect it shares with 616 and 618.
        616  step  3.0  past median 19.8  ratio 0.15   dropped
        618  step  4.2  past median 16.5  ratio 0.26   dropped
        621  step  3.6  past median 15.2  ratio 0.24   dropped
        every other flight frame                       ratio 0.59 to 2.91, kept
    `MIN_MOVE` keeps the rule quiet where the ball is genuinely still: before the kick a small
    step is the truth, not an error, and a ratio has no meaning when the yardstick is zero.
    This also covers the 41 rows that repeat the previous pixel exactly (ratio 0 where moving);
    the ones that survive are repeats of a ball that really was standing still.

    Not dropped: the 14 boxes <= 4 px wide (frames 29-34, 62-68, 100, 101, 134). Every one is in
    the first five seconds and none is inside the flight window, so a size rule would change no
    number in this stage -- and deciding whether those are the match ball or a spare by the ad
    boards means looking at 14 crops, not picking a threshold.
    """
    # ponytail: one global ratio. If a clip has a slow-motion replay in it the step scale changes
    # mid-clip and the past window absorbs it -- but a genuine re-time (a cut) will drop a frame.
    steps = {}
    for previous, row in zip(rows, rows[1:]):
        if row["frame"] == previous["frame"] + 1:
            steps[row["frame"]] = float(np.linalg.norm(
                np.array(row["px"], dtype=float) - np.array(previous["px"], dtype=float)))
    for row in rows:
        step = steps.get(row["frame"])
        recent = [steps[f] for f in range(row["frame"] - WINDOW, row["frame"]) if f in steps]
        if step is None or len(recent) < 3:
            continue
        scale = float(np.median(recent))
        if scale >= MIN_MOVE and step < RATIO * scale:
            row["source"] = "dropped"
    return rows


def main(clip):
    data = json.loads((ROOT / "data" / "ball" / f"{clip}.json").read_text())
    rows = [dict(r, source="label") for r in data["frames"]]
    rows = suspicious(fill_gaps(rows, data["missing"]))
    out = ROOT / "data" / "ball" / f"{clip}_clean.json"
    out.write_text(json.dumps({**data, "frames": rows}))
    counts = {k: sum(r["source"] == k for r in rows) for k in ("label", "filled", "dropped")}
    print(f"wrote {out}: {counts}")
    usable = [r for r in rows if r["source"] == "label" and FLIGHT[0] <= r["frame"] <= FLIGHT[1]]
    print(f"  frames the B fit can use ({FLIGHT[0]}-{FLIGHT[1]}): {len(usable)} of {FLIGHT[1] - FLIGHT[0] + 1}")


def demo():
    """Properties a correct clean must have. These do not tell you HOW -- only what can't be true."""
    data = json.loads((ROOT / "data" / "ball" / "SNGS-043.json").read_text())
    rows = suspicious(fill_gaps([dict(r, source="label") for r in data["frames"]], data["missing"]))
    by_frame = {r["frame"]: r for r in rows}

    assert [r["frame"] for r in rows] == sorted(r["frame"] for r in rows), "still sorted by frame"

    # 1. Nothing invented inside the flight window: every frame the fit uses is a real measurement
    #    or is left out, never filled. All three gaps are pre-kick, so this costs you nothing.
    flight = [r for r in rows if FLIGHT[0] <= r["frame"] <= FLIGHT[1]]
    assert all(r["source"] != "filled" for r in flight), "a filled frame ended up inside the flight"

    # 2. A filled frame has to sit between the labels it was built from. Gap 367 is the easy case.
    if 367 in by_frame:
        x = by_frame[367]["px"][0]
        assert by_frame[366]["px"][0] < x < by_frame[368]["px"][0], f"367 is not between its neighbours: {x}"

    # 3. The three mistimed frames go, and NOTHING else in the flight does. Counting survivors is
    #    not enough: a rule that drops no flight frame at all also leaves 21 of them standing.
    dropped = {r["frame"] for r in flight if r["source"] == "dropped"}
    assert dropped == {616, 618, 621}, f"wrong flight frames dropped: {sorted(dropped)}"

    # 4. The shot is still there to fit, and frame 602 (the kick) is not negotiable.
    kept = [r for r in flight if r["source"] == "label"]
    assert len(kept) >= 15, f"only {len(kept)} usable flight frames left"
    assert by_frame[602]["source"] == "label", "frame 602 is the kick, it cannot be dropped"

    print(f"clean self-check ok: {len(kept)} usable flight frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="SNGS-043")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    demo() if args.self_check else main(args.clip)
