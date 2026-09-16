"""Stage 2: the foot pixel of each tracked box -> pitch meters, with inv(H) from Stage 1.

Usage: uv run python src/track/to_pitch.py clip04 [smoothing window in frames]   (or SNGS-028)
  Without the window it smooths over SMOOTH_S seconds; 0 turns smoothing off.
Reads data/track/<clip>_football-player-detection-v9_botsort.json and data/camera/<clip>.json.
Writes data/tracks/<clip>.json (tracks.json, format in PLAN.md -> Data formats).
Prints two checks:
  1. your clicked pitch points (annotations/clicks) -> meters, vs where they really are.
     Tests pixels_to_meters alone.
  2. the top speed of each player in m/s. Tests everything together: a sprinting footballer
     tops out around 10 m/s, so anything far above that is a bug, a bad H, or an ID switch.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from homography import load_pairs, project  # noqa: E402  (lives in src/calib)
from clips import fps  # noqa: E402  (same folder)

# Drop person tracks whose average YOLO confidence is below this: on clip04 that removes a pile of
# towels by the post (5 IDs) and a steward behind the goal, all 0.20-0.43, while every real player
# is 0.60 or more. Picked on clip04 alone: check the gap again on other clips.
MIN_CONF = 0.5
# Smoothing window per track, in SECONDS (frames would mean different amounts of time per clip:
# SoccerNet runs at 25 fps, clip04 at 49.95). Measured on SNGS-028 + SNGS-043 against the ground
# truth: the position error keeps falling up to ~1 s, and at 0.84 s our players' 95% speed (5.5 m/s)
# matches the true players' (5.3 m/s). Longer flattens real turns: at 2 s the error gets worse again.
SMOOTH_S = 0.84
# Joining track pieces: how long a player may be lost, and how far the pieces may be apart. Starting
# guesses, to be tuned against (C) in eval_soccernet.py (with (B) as the guard: too generous and we
# glue two different players together, which makes the position error worse).
JOIN_GAP_S, JOIN_DIST = 1.0, 3.0


def smooth_tracks(frames, window):
    """Average each track's positions over `window` frames, to take the wobble out.

    frames: the list that goes into tracks.json: [{"frame": 0, "players": [{"id", "team", "x", "y",
            "visible"}, ...], "ball_px": ...}, ...]. Frame numbers go up but can have gaps.
    window: how many frames to average over, centred on each point (odd numbers are easiest: 5 means
            the point itself plus 2 before and 2 after). window <= 1 means no smoothing.
    Returns: frames with the x and y of every player replaced by the average of that SAME id's
             positions in the surrounding frames. Don't mix two ids together, and don't average
             across a long gap in a track (a track that stops at frame 10 and comes back at 200).
    Careful: the ends of a track have no neighbours on one side.
    """
    frames = [dict(fr, players=[dict(p) for p in fr["players"]]) for fr in frames]  # deep copy
    if window <= 1:
        return frames
    half = window // 2
    tracks = {}  # id -> list of (frame, x, y, the player dict itself: writing into it updates `frames`)
    for fr in frames:
        for p in fr["players"]:
            tracks.setdefault(p["id"], []).append((fr["frame"], p["x"], p["y"], p))
    for tid, t in tracks.items():
        pos = np.array([row[:3] for row in t])  # (frame, x, y) as it was BEFORE smoothing
        for i, (f, x, y, p) in enumerate(t):
            # Find the surrounding frames of this same id, within the window.
            # Don't average across a long gap in a track (a track that stops at frame 10 and comes back at 200).
            start = max(0, i - half)
            while start < i and pos[start, 0] < f - half:
                start += 1
            end = min(len(t), i + half + 1)
            while end > i + 1 and pos[end - 1, 0] > f + half:
                end -= 1
            if end - start > 1:  # otherwise no neighbours to average with
                p["x"] = float(np.mean(pos[start:end, 1]))
                p["y"] = float(np.mean(pos[start:end, 2]))
    return frames


def check_smooth_tracks():
    """Two cases with a known answer, run before smoothing anything."""
    # A player running straight at a steady speed: a centred average changes nothing in the middle.
    straight = [{"frame": f, "players": [{"id": 1, "team": None, "x": 0.5 * f, "y": 2.0, "visible": True}],
                 "ball_px": None} for f in range(11)]
    out = smooth_tracks([dict(fr, players=[dict(p) for p in fr["players"]]) for fr in straight], 5)
    mid = [p for fr in out[3:8] for p in fr["players"]]
    assert np.allclose([p["x"] for p in mid], [0.5 * f for f in range(3, 8)]), "a straight run must survive smoothing"
    # One point knocked 1 m sideways: smoothing must pull it back towards the line.
    bumpy = [dict(fr, players=[dict(p) for p in fr["players"]]) for fr in straight]
    bumpy[5]["players"][0]["y"] = 3.0
    out = smooth_tracks(bumpy, 5)
    assert abs(out[5]["players"][0]["y"] - 2.0) < 0.5, "a single bad point must be pulled back"


def join_tracks(frames, max_gap, max_dist):
    """Glue track pieces back together: a track that stops and another that starts nearby = one player.

    The tracker loses a player behind another one and gives them a new id when they come back. On
    SNGS-028 the 23 real people got 190 of our ids. Joining happens in METERS, not pixels, because
    the camera pans: a player standing still moves hundreds of pixels but zero meters.

    frames: as in smooth_tracks (already smoothed when main calls this).
    max_gap: how many frames a player may be lost for and still count as the same player.
    max_dist: how far apart (meters) the end of one piece and the start of the next may be.
    Returns: frames where the joined pieces all carry the FIRST piece's id. Positions don't change,
             only ids: this can't help (B), only (C).
    Rules:
      - Piece B can follow piece A only if B starts AFTER A ends (they must not overlap in time:
        if both are on screen in the same frame they are two different people).
      - One-to-one, like match_frame: a piece has at most one follower and at most one predecessor.
        Closest first is the safe order here too.
      - A chain A -> B -> C must all end up with A's id.
    (Claude wrote this on my request, walked through line by line.)
    """
    frames = [dict(fr, players=[dict(p) for p in fr["players"]]) for fr in frames]  # deep copy
    pieces = {}  # our id -> where and when that piece starts and ends
    for fr in frames:
        for p in fr["players"]:
            piece = pieces.setdefault(p["id"], {"first": fr["frame"], "start": (p["x"], p["y"])})
            piece["last"], piece["end"] = fr["frame"], (p["x"], p["y"])  # overwritten until the last frame

    # Every pair that COULD be the same player: j starts after i ends, soon enough and near enough.
    # ponytail: every piece against every piece (200 x 200 here). Bucket by frame if a clip gets long.
    candidates = []
    for i, a in pieces.items():
        for j, b in pieces.items():
            gap = b["first"] - a["last"]
            if 0 < gap <= max_gap:  # gap > 0 means they never share a frame
                dist = float(np.hypot(*(np.array(b["start"]) - np.array(a["end"]))))
                if dist <= max_dist:
                    candidates.append((dist, i, j))

    follows = {}  # piece -> the piece it continues; following the chain up gives the player's first id

    def first_piece(i):
        while i in follows:
            i = follows[i]
        return i

    has_next, has_prev = set(), set()
    for dist, i, j in sorted(candidates):  # closest first, like match_frame
        if i in has_next or j in has_prev or first_piece(i) == first_piece(j):
            continue  # i already continues somewhere, j already follows something, or it's a loop
        has_next.add(i)
        has_prev.add(j)
        follows[j] = i

    for fr in frames:
        for p in fr["players"]:
            p["id"] = first_piece(p["id"])
    return frames


def check_join_tracks():
    """Three cases with a known answer."""
    def clip(pieces):  # pieces: {id: [(frame, x, y), ...]} -> frames, as in tracks.json
        fs = sorted({f for t in pieces.values() for f, _, _ in t})
        return [{"frame": f, "ball_px": None,
                 "players": [{"id": i, "team": None, "x": x, "y": y, "visible": True}
                             for i, t in pieces.items() for g, x, y in t if g == f]} for f in fs]

    def ids(frames):
        return {p["id"] for fr in frames for p in fr["players"]}

    # Same player: piece 1 ends at frame 4 at x = 4, piece 2 starts at frame 9 at x = 5.
    same = clip({1: [(f, float(f), 0.0) for f in range(5)], 2: [(f, 5.0, 0.0) for f in range(9, 14)]})
    assert len(ids(join_tracks(same, 10, 3))) == 1, "a short gap, a short distance: one player"
    # Two people on screen at the same time are never the same player, however close they are.
    both = clip({1: [(f, 0.0, 0.0) for f in range(10)], 2: [(f, 1.0, 0.0) for f in range(10)]})
    assert len(ids(join_tracks(both, 10, 3))) == 2, "pieces that overlap in time are two people"
    # Far apart in meters: leave them alone.
    far = clip({1: [(f, 0.0, 0.0) for f in range(5)], 2: [(f, 20.0, 0.0) for f in range(9, 14)]})
    assert len(ids(join_tracks(far, 10, 3))) == 2, "20 m apart is not the same player"


def foot_point(xyxy):
    """Box [x1, y1, x2, y2] in pixels -> the foot pixel [u, v].

    Remember: in images, v grows DOWNWARD (row 0 is the top of the picture).
    """
    return np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]])  # middle of the bottom edge


def pixels_to_meters(H, pixels):
    """Pixels -> pitch meters. Only right for points on the grass (z = 0).

    H: (3, 3) array from camera.json. It maps pitch meters -> pixels.
    pixels: (N, 2) array, rows [u, v].
    Returns: (N, 2) array, rows [x, y] in pitch meters.
    """
    return project(np.linalg.inv(H), pixels)  # same maths as meters -> pixels, just the inverse H


def frame_entry(frame, boxes, H):
    """One frame of tracks.json (format: PLAN.md -> Data formats).

    frame: frame number (int).
    boxes: this frame's boxes from the raw tracking file, dicts with "id", "cls", "conf",
           "xyxy". cls is "ball" for the ball, anything else is a person.
    H: (3, 3) array for this frame, or None if PnLCalib found no camera.
    Returns: {"frame": frame,
              "players": [{"id": 7, "team": None, "x": -3.2, "y": 14.8, "visible": True,
                           "box_px": [x1, y1, x2, y2]}, ...],
              "ball_px": [u, v] or None}
      - players: every person box, placed with foot_point + pixels_to_meters.
        team = None until teams.py fills it in.
        box_px = the box this dot came from. Kept because join_tracks renames ids, so the raw
        tracking file can no longer be looked up by id, and teams/pose need the crop anyway.
        visible = True: every box is something YOLO saw (later, filled-in gaps get False).
        If H is None we can't place anyone: players = [].
      - ball_px: the centre of the ball box in pixels (the most confident one if there are
        several), or None if there's no ball in this frame.
    """
    players = []
    if H is not None:  # no camera -> we can't place anyone
        for b in boxes:
            if b["cls"] != "ball":
                foot_px = foot_point(b["xyxy"])
                x, y = pixels_to_meters(H, np.array([foot_px]))[0]
                players.append({"id": b["id"], "team": None, "x": x, "y": y, "visible": True,
                                "box_px": [round(v, 1) for v in b["xyxy"]]})

    # The ball stays in pixels, so it needs no H. Several ball boxes -> keep the most confident.
    balls = [b for b in boxes if b["cls"] == "ball"]
    ball_px = None
    if balls:
        x1, y1, x2, y2 = max(balls, key=lambda b: b["conf"])["xyxy"]
        ball_px = [(x1 + x2) / 2, (y1 + y2) / 2]
    return {"frame": int(frame), "players": players, "ball_px": ball_px}


def main(clip, window=None):
    window = round(SMOOTH_S * fps(clip)) if window is None else int(window)  # 0 or 1 = no smoothing
    raw = json.load(open(ROOT / "data" / "track" / f"{clip}_football-player-detection-v9_botsort.json"))
    cams = {c["frame"]: np.array(c["H"])
            for c in json.load(open(ROOT / "data" / "camera" / f"{clip}.json"))["frames"]}

    # Check 1: known pitch points. A clicked pixel -> meters should land on the pitch model point.
    errs = []
    for path in sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json")):
        frame = int(path.stem.split("_")[1])
        _, pitch_xy, pixels = load_pairs(path)
        errs += list(np.linalg.norm(pixels_to_meters(cams[frame], pixels) - pitch_xy, axis=1))
    if errs:  # SoccerNet clips have no clicks: eval_soccernet.py checks them against their labels
        print(f"check 1, clicks -> meters: mean {np.mean(errs):.2f} m, worst {np.max(errs):.2f} m ({len(errs)} points)")

    # A ball track sometimes gets a person label for a few frames: call the whole track "ball".
    ball_ids = {b["id"] for f in raw["frames"] for b in f["boxes"] if b["cls"] == "ball"}
    confs = {}
    for f in raw["frames"]:
        for b in f["boxes"]:
            confs.setdefault(b["id"], []).append(b["conf"])
    unsure = {i for i, c in confs.items() if i not in ball_ids and np.mean(c) < MIN_CONF}
    print(f"dropped {len(unsure)} person tracks with mean confidence < {MIN_CONF}: {sorted(unsure)}")
    frames = []
    for f in raw["frames"]:
        boxes = [dict(b, cls="ball") if b["id"] in ball_ids else b for b in f["boxes"] if b["id"] not in unsure]
        frames.append(frame_entry(f["frame"], boxes, cams.get(f["frame"])))
    if int(window) > 1:
        smoothed = smooth_tracks(frames, int(window))
        if smoothed is None:
            print("smooth_tracks isn't written yet (TODO(human)): saving the raw positions")
        else:
            check_smooth_tracks()
            frames = smoothed
            print(f"smoothed each track over {window} frames ({int(window) / fps(clip):.2f} s)")

    joined = join_tracks(frames, round(JOIN_GAP_S * fps(clip)), JOIN_DIST)
    if joined is None:
        print("join_tracks isn't written yet (TODO(human)): keeping the tracker's ids")
    else:
        check_join_tracks()
        before = len({p["id"] for fr in frames for p in fr["players"]})
        frames = joined
        after = len({p["id"] for fr in frames for p in fr["players"]})
        print(f"joined track pieces (gap <= {JOIN_GAP_S} s, <= {JOIN_DIST} m apart): {before} ids -> {after}")

    out = ROOT / "data" / "tracks" / f"{clip}.json"
    out.parent.mkdir(exist_ok=True)
    FPS = fps(clip)
    json.dump({"clip": clip, "fps": FPS, "frames": frames}, open(out, "w"))
    print(f"{len(frames)} frames -> {out}")

    tracks = {}  # id -> list of (frame, x, y), read back from what goes into tracks.json
    for fr in frames:
        for p in fr["players"]:
            tracks.setdefault(p["id"], []).append((fr["frame"], p["x"], p["y"]))

    # Check 2: top speed per player. Measured every 10th position (0.2 s): frame-to-frame, a 10 cm
    # wobble of the box would already read as 5 m/s.
    print("check 2, top speed per ID:")
    for tid, t in sorted(tracks.items(), key=lambda kv: -len(kv[1])):
        n = len(t)
        t = np.array(t)[::10]
        if len(t) < 2:
            continue
        speed = np.linalg.norm(np.diff(t[:, 1:], axis=0), axis=1) / (np.diff(t[:, 0]) / FPS)
        i = speed.argmax()
        print(f"  #{tid:>3}: {n:>3} frames, starts at ({t[0, 1]:6.1f}, {t[0, 2]:5.1f}) m, "
              f"top {speed[i]:5.1f} m/s around frame {int(t[i + 1, 0])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
