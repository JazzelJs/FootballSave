"""Stage 2: position error in meters on a SoccerNet GSR clip, against its ground-truth labels.

Usage: uv run python src/track/eval_soccernet.py SNGS-028
Reads data/soccernet/<clip>/Labels-GameState.json and data/camera/<clip>.json.

(A) camera only: SoccerNet's own (perfect) boxes -> foot pixel -> OUR H -> meters, vs the true
    meters. Detection and tracking play no part, so this is PnLCalib's error in meters.
(B) full pipeline: our dots from data/tracks/<clip>.json (detect_track.py + to_pitch.py), matched
    one-to-one to the true players in each frame. Also counts true players we missed and extra dots.
(C) identity: which of our track ids sat on each true player, over the whole clip: how many
    different ids one player got (fragments) and how often the id changed (switches). (B) measures
    where a dot is, (C) measures whether it keeps the same name. Joining track pieces is judged here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
from compare_pnl import GOAL_SIDE, pitch_to_soccernet  # noqa: E402  (lives in src/calib)
from to_pitch import pixels_to_meters  # noqa: E402

FPS = 25
PERSON = {1, 2, 3}  # SoccerNet category ids: player, goalkeeper, referee (4 = ball, 5 = pitch lines)
MAX_DIST = 3.0  # m: a dot further than this from every true player doesn't count as finding them
# Last frame of actual football in a clip. After the goal in SNGS-043 the players celebrate on the
# grass, and the detector (trained on players playing) loses a third of them: report both.
PLAY_END = {"SNGS-043": 634}


def load_truth(clip):
    """frame -> (foot_px (N, 2), true_xy (N, 2) SoccerNet meters, raw_xy (N, 2), track_id (N,)).

    track_id is the labels' own id for that person, the same number in every frame: what our track
    ids should look like if detection and tracking were perfect (24 people in SNGS-043).
    """
    d = json.load(open(ROOT / "data" / "soccernet" / clip / "Labels-GameState.json"))
    frame_of = {im["image_id"]: int(im["file_name"].split(".")[0]) for im in d["images"]}  # 000001.jpg -> 1
    per_frame = {}
    for a in d["annotations"]:
        if a["category_id"] not in PERSON or a["bbox_pitch"] is None:
            continue  # a handful of people have no pitch position (off the pitch)
        b, p, r = a["bbox_image"], a["bbox_pitch"], a["bbox_pitch_raw"]
        foot = (b["x"] + b["w"] / 2, b["y"] + b["h"])  # middle of the bottom edge, like foot_point
        raw = (r["x_bottom_middle"], r["y_bottom_middle"]) if r else (np.nan, np.nan)  # raw is sometimes missing
        per_frame.setdefault(frame_of[a["image_id"]], []).append(
            (foot, (p["x_bottom_middle"], p["y_bottom_middle"]), raw, a["track_id"]))
    # (the track ids come back as floats; they are only ever compared, never used as numbers)
    return {f: tuple(np.array(col, float) for col in zip(*rows)) for f, rows in per_frame.items()}


def camera_errors(H, foot_px, true_xy, goal):
    """Error in meters of each true foot pixel pushed through OUR camera.

    H: (3, 3) from camera.json, OUR pitch meters -> pixels.
    foot_px: (N, 2) true foot pixels [u, v] from the labels.
    true_xy: (N, 2) true positions in SoccerNet meters (X, Y): origin centre spot.
    goal: "left" / "right", for pitch_to_soccernet.
    Returns: (N,) distances in meters.
    Careful: our H gives OUR frame, the labels are in SoccerNet's. Both functions you need exist.
    """
    ours = pixels_to_meters(H, foot_px)  # (N, 2) OUR meters: origin = centre of the attacked goal line
    soccer = pitch_to_soccernet(ours, goal)[:, :2]  # (N, 2) SoccerNet X, Y (drop Z, it's 0 on the grass)
    return np.linalg.norm(soccer - true_xy, axis=1)  # straight-line distance per person


def load_ours(clip, goal):
    """frame -> (our track ids (N,), our dots (N, 2) in SoccerNet meters), from tracks.json."""
    tracks = json.load(open(ROOT / "data" / "tracks" / f"{clip}.json"))
    return {fr["frame"]: (np.array([p["id"] for p in fr["players"]]),
                          pitch_to_soccernet(np.array([[p["x"], p["y"]] for p in fr["players"]]).reshape(-1, 2), goal)[:, :2])
            for fr in tracks["frames"]}


def match_frame(ours, true, max_dist):
    """Pair our dots with the true players in ONE frame, one-to-one, and score it.

    ours: (N, 2) our dots, SoccerNet meters (X, Y). N can be 0.
    true: (M, 2) the true players, SoccerNet meters (X, Y).
    max_dist: meters. A pair further apart than this is not a match.
    Returns (errors, missed, extra, matched):
      errors: (K,) array, the distance in meters of each matched pair (K = number of pairs)
      missed: int, true players with no dot (M - K)
      extra: int, dots with no true player (N - K)
      matched: list of (dot index, true player index), so the caller can see WHICH dot went with
               which player. Only used by the identity numbers (Claude added this part).
    One-to-one: one dot can be the match of at most one true player, and the other way round.
    """
    errors = []
    matched = []
    missed = 0
    extra = 0
    if len(ours) == 0:
        missed = len(true)
        extra = 0
        return np.array(errors), missed, extra, matched
    if len(true) == 0:
        missed = 0
        extra = len(ours)
        return np.array(errors), missed, extra, matched

    # Greedy matching: closest pairs first, skip pairs whose dot or player is already used.
    # (Claude wrote this part on my request.)
    pairs = sorted((np.linalg.norm(ours[i] - true[j]), i, j) for i in range(len(ours)) for j in range(len(true)))
    used_dots, used_true = set(), set()
    for dist, i, j in pairs:
        if dist > max_dist:
            break  # the list is sorted: every pair after this one is even further apart
        if i in used_dots or j in used_true:
            continue  # a closer pair already took this dot or this player
        errors.append(dist)
        matched.append((i, j))
        used_dots.add(i)
        used_true.add(j)
    missed = len(true) - len(errors)
    extra = len(ours) - len(errors)
    return np.array(errors), missed, extra, matched


def check_match_frame():
    """Tiny cases with a known answer. Any correct one-to-one matching passes them."""
    e, m, x, _ = match_frame(np.array([[0, 0], [10, 0], [50, 50]]), np.array([[0.5, 0], [10, 1], [30, 30], [31, 30]]), 3)
    assert np.allclose(sorted(e), [0.5, 1.0]) and (m, x) == (2, 1), "2 pairs, 2 true players missed, 1 extra dot"
    e, m, x, _ = match_frame(np.array([[0, 0], [0.2, 0]]), np.array([[0.1, 0]]), 3)
    assert len(e) == 1 and (m, x) == (0, 1), "two dots on one player: only one of them is a match, the other is extra"
    e, m, x, _ = match_frame(np.zeros((0, 2)), np.array([[0, 0]]), 3)
    assert len(e) == 0 and (m, x) == (1, 0), "no dots at all: everyone missed"
    # A at X = 0, B at X = 2.5; dots at X = -2.8 and 1.5. If A goes first it takes the 1.5 dot and B
    # is left with -2.8 (5.3 m away): 1 pair. The right answer pairs A-(-2.8) and B-1.5: 2 pairs.
    e, m, x, _ = match_frame(np.array([[-2.8, 0], [1.5, 0]]), np.array([[0, 0], [2.5, 0]]), 3)
    assert np.allclose(sorted(e), [1.0, 2.8]) and (m, x) == (0, 0), \
        "A must not steal B's dot: the order you match in matters (see the A/B example)"


def summary(name, errs):
    errs = np.asarray(errs)
    return (f"{name:<14} mean {errs.mean():5.2f} m  median {np.median(errs):5.2f}  "
            f"95% {np.percentile(errs, 95):5.2f}  max {errs.max():6.2f}  ({len(errs)} people)")


def main(clip):
    goal = GOAL_SIDE[clip]
    truth = load_truth(clip)
    cam_frames = json.load(open(ROOT / "data" / "camera" / f"{clip}.json"))["frames"]
    cams = {c["frame"]: np.array(c["H"]) for c in cam_frames}
    filled = {c["frame"] for c in cam_frames if c["filled"]}
    no_cam = sorted(set(truth) - set(cams))
    print(f"{clip}: {len(truth)} labelled frames, {len(filled)} with a filled-in camera, {len(no_cam)} without one (skipped)")

    # The labels' own wobble: SoccerNet stores two positions per person (bbox_pitch and bbox_pitch_raw).
    # How far apart they are = how exact the "truth" itself is. Don't expect to beat this.
    floor = np.concatenate([np.linalg.norm(t - r, axis=1) for _, t, r, _ in truth.values()])
    print(summary("labels' own", floor[~np.isnan(floor)]))

    per_frame = {f: camera_errors(cams[f], foot, true, goal) for f, (foot, true, _, _) in truth.items() if f in cams}
    all_errs = np.concatenate(list(per_frame.values()))
    print(summary("(A) camera", all_errs))
    for name, fs in [("  PnLCalib", set(per_frame) - filled), ("  filled in", filled & set(per_frame))]:
        if fs:
            print(summary(name, np.concatenate([per_frame[f] for f in fs])))
    if np.median(all_errs) > 20:
        print("  median > 20 m: that's a mirrored or wrong-end pitch, check GOAL_SIDE / pitch_to_soccernet")

    print("(A) per 5 s:")
    for start in range(1, 751, 5 * FPS):
        chunk = [e for f, e in per_frame.items() if start <= f < start + 5 * FPS]
        if chunk:
            print(f"  frames {start:3d}-{start + 5 * FPS - 1:3d}  " + summary("", np.concatenate(chunk)).strip())
    worst = sorted(per_frame, key=lambda f: -per_frame[f].mean())[:5]
    print("(A) worst frames (mean m):", ", ".join(f"{f}: {per_frame[f].mean():.1f}" for f in worst))

    # (B) full pipeline: our dots vs the true players, frame by frame.
    if match_frame(np.zeros((1, 2)), np.zeros((1, 2)), MAX_DIST) is None:
        print("\n(B): write match_frame first (TODO(human) above)")
        return
    check_match_frame()
    ours = load_ours(clip, goal)
    empty = (np.zeros(0), np.zeros((0, 2)))
    scores = {f: match_frame(ours.get(f, empty)[1], true, MAX_DIST) for f, (_, true, _, _) in truth.items()}
    def report(name, fs):
        errs = np.concatenate([scores[f][0] for f in fs])
        n = sum(len(truth[f][1]) for f in fs)
        missed, extra = sum(scores[f][1] for f in fs), sum(scores[f][2] for f in fs)
        print(summary(name, errs))
        print(f"  missed {missed}/{n} true players ({100 * missed / n:.0f}%), "
              f"{extra} extra dots ({extra / len(fs):.1f} per frame), pairs further than {MAX_DIST} m don't count")
        return errs

    print()
    errs_b = report("(B) pipeline", sorted(scores))
    if clip in PLAY_END:  # the same, but only while football is being played
        report(f"(B) to {PLAY_END[clip]}", [f for f in sorted(scores) if f <= PLAY_END[clip]])
    print(f"  (B) - (A), median: {np.median(errs_b) - np.median(all_errs):+.2f} m = what detection + tracking add")
    worst = sorted(scores, key=lambda f: -(scores[f][0].mean() if len(scores[f][0]) else 0))[:5]
    print("(B) worst frames (mean m):", ", ".join(f"{f}: {scores[f][0].mean():.1f}" for f in worst))
    most_missed = sorted(scores, key=lambda f: -scores[f][1])[:5]
    print("(B) most missed:", ", ".join(f"{f}: {scores[f][1]}/{len(truth[f][1])}" for f in most_missed))

    # (C) identity: (B) says WHERE, this says WHO. Follow each true player through the clip and look
    # at which of our track ids was on them: how many different ones (fragments), and how often it
    # changed (switches). Perfect tracking = 1 fragment and 0 switches per player.
    seen = {}  # true track id -> [(frame, our id), ...] in frame order
    for f in sorted(scores):
        our_ids = ours.get(f, empty)[0]
        for i, j in scores[f][3]:
            seen.setdefault(truth[f][3][j], []).append((f, our_ids[i]))
    frags = np.array([len({i for _, i in v}) for v in seen.values()])
    switches = np.array([sum(a[1] != b[1] for a, b in zip(v, v[1:])) for v in seen.values()])
    covered = np.array([len(v) for v in seen.values()])
    print()
    print(f"(C) identity: {len(seen)} true people, we gave them {frags.sum()} different track ids "
          f"({frags.mean():.1f} each, worst {frags.max()})")
    print(f"  ID switches: {switches.sum()} in total, {switches.mean():.1f} per player "
          f"(one switch per {covered.sum() / max(switches.sum(), 1):.0f} frames of following someone)")
    print(f"  our IDs in tracks.json: {len({i for f in ours for i in ours[f][0]})} "
          f"(the extra ones are dots that never matched anybody)")
    # The other way round: one of OUR ids sitting on several true people. Joining track pieces too
    # greedily glues two players into one id, which LOWERS the fragment count while being wrong:
    # this is the number that catches it, not (B) (joining never moves a dot).
    people_of = {}
    for t, v in seen.items():
        for _, i in v:
            people_of.setdefault(i, set()).add(t)
    merged = {i: p for i, p in people_of.items() if len(p) > 1}
    print(f"  merged ids (one id on several people): {len(merged)}/{len(people_of)}, "
          f"worst sits on {max((len(p) for p in merged.values()), default=1)} people")
    worst_id = sorted(seen, key=lambda t: -len({i for _, i in seen[t]}))[:5]
    print("  most broken-up players:", ", ".join(
        f"#{int(t)}: {len({i for _, i in seen[t]})} ids over {len(seen[t])} frames" for t in worst_id))


if __name__ == "__main__":
    main(sys.argv[1])
