"""C3 + C4: look at the fitted flight, instead of trusting the numbers.

Usage: uv run python src/ball/draw.py SNGS-043            # one jpg at the kick + the mp4 + the plot
       uv run python src/ball/draw.py SNGS-043 --frame 612

C4 draws the fit back onto the broadcast frames, the same way src/pose/draw_facing.py does:
    red    where the label says the ball is (the measurement)
    green  where the fit puts it, through that frame's own camera
    yellow the whole fitted flight, projected
    grey   the flight's ground shadow -- the gap between grey and yellow IS the height
C3 plots our shadow against the labels' `bbox_pitch`, which is their own ground shadow: the two
must agree before the kick, and separate afterwards by an amount that grows with the height.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "calib"))
sys.path.insert(0, str(ROOT / "src" / "track"))
sys.path.insert(0, str(ROOT / "src" / "ball"))
from compare_pnl import GOAL_SIDE, camera_matrix, pitch_to_soccernet, project_3d  # noqa: E402
from clips import frame_path  # noqa: E402
import fit as ballfit  # noqa: E402

PAD = 8  # frames drawn either side of the fit window, so the ends are visible


def dense_path(r, extra=0.25):
    """The fitted flight sampled finely, a bit past the window, for drawing a smooth curve."""
    times = np.arange(0.0, r["times"][-1] + extra, 1.0 / (r["fps"] * 8))
    return times, ballfit.path(np.array(r["p0"]), np.array(r["v0"]), times, r["k"])


def to_px(clip, P, points, on_ground=False):
    soccer = pitch_to_soccernet(points[:, :2], GOAL_SIDE[clip])
    soccer[:, 2] = 0.0 if on_ground else -points[:, 2]
    return project_3d(P, soccer)


def overlay(clip, r, cameras, frame, ball_rows):
    image = cv2.imread(str(frame_path(clip, frame)))
    if image is None or frame not in cameras:
        return None
    P = cameras[frame]
    times, points = dense_path(r)
    for pts, colour, thickness in ((to_px(clip, P, points, on_ground=True), (170, 170, 170), 2),
                                   (to_px(clip, P, points), (0, 220, 255), 2)):
        cv2.polylines(image, [pts.astype(np.int32)], False, colour, thickness)
    if frame in r["frames"]:
        i = r["frames"].index(frame)
        point = ballfit.path(np.array(r["p0"]), np.array(r["v0"]),
                             [r["times"][i]], r["k"])
        fx, fy = to_px(clip, P, point)[0].astype(int)
        ox, oy = np.array(r["observed"][i]).astype(int)
        cv2.circle(image, (ox, oy), 9, (0, 0, 255), 2)
        cv2.circle(image, (fx, fy), 5, (0, 220, 0), -1)
        cv2.line(image, (ox, oy), (fx, fy), (0, 220, 0), 1)
        cv2.putText(image, f"{r['px_per_frame'][i]:.1f} px", (ox + 12, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        height = float(point[0][2])
        cv2.putText(image, f"z = {height:.2f} m", (ox + 12, oy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    elif frame in ball_rows:
        ox, oy = np.array(ball_rows[frame]["px"]).astype(int)
        cv2.circle(image, (ox, oy), 9, (0, 0, 255), 1)
    cv2.putText(image, f"frame {frame}   {r['speed_kmh']:.0f} km/h   {r['px_median']:.1f} px median",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image


def shadow_plot(clip, r, ball_rows):
    """C3: their ground shadow vs ours, and the height that explains the gap."""
    frames = [s["frame"] for s in r["shadow"]]
    gaps = [s["gap_m"] for s in r["shadow"]]
    heights = [s["z"] for s in r["shadow"]]
    before = sorted(f for f in ball_rows if r["kick"] - 12 <= f < r["kick"] and ball_rows[f]["pitch_px"])
    times, points = dense_path(r)
    ours = pitch_to_soccernet(points[:, :2], GOAL_SIDE[clip])

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    top.plot([ball_rows[f]["pitch_px"][0] for f in before],
             [ball_rows[f]["pitch_px"][1] for f in before], "o-", color="0.4",
             label="labels' shadow, before the kick (ball really is on the grass)")
    top.plot([ball_rows[f]["pitch_px"][0] for f in frames],
             [ball_rows[f]["pitch_px"][1] for f in frames], "o-", color="tab:red",
             label="labels' shadow, during the flight")
    top.plot(ours[:, 0], ours[:, 1], "-", color="tab:blue", label="our fit, on the grass (x, y)")
    top.axvline(52.5, color="k", ls="--", lw=1)
    top.text(52.6, top.get_ylim()[0], " goal line", fontsize=8)
    top.set_xlabel("SoccerNet X (m)"), top.set_ylabel("SoccerNet Y (m)")
    top.legend(fontsize=8), top.set_title(f"{clip}: ground track, ours vs the labels' shadow")

    bottom.plot(frames, gaps, "o-", color="tab:red", label="gap between the two shadows (m)")
    bottom.plot(frames, heights, "o-", color="tab:blue", label="our fitted height z (m)")
    bottom.axvline(r["kick"], color="k", ls="--", lw=1)
    bottom.text(r["kick"] + 0.2, max(gaps) * 0.9, " kick", fontsize=8)
    bottom.set_xlabel("frame"), bottom.legend(fontsize=8)
    bottom.set_title("a shadow is only the ball while the ball is on the grass")
    figure.tight_layout()
    out = ROOT / "outputs" / f"ball_shadow_{clip}.png"
    figure.savefig(out, dpi=120), plt.close(figure)
    return out


def main(clip, single=None):
    r = json.loads((ROOT / "data" / "ball" / f"{clip}_fit.json").read_text())
    raw = json.loads((ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json").read_text())
    cameras = {f["frame"]: camera_matrix(f["cam_params"]) for f in raw["frames"] if f["ok"]}
    ball = json.loads((ROOT / "data" / "ball" / f"{clip}_clean.json").read_text())
    ball_rows = {row["frame"]: row for row in ball["frames"] if row["source"] == "label"}

    if single is not None:
        image = overlay(clip, r, cameras, single, ball_rows)
        out = ROOT / "outputs" / f"ball_fit_{clip}_{single:06d}.jpg"
        cv2.imwrite(str(out), image)
        print(f"wrote {out}")
        return

    stills = ROOT / "outputs" / f"ball_fit_{clip}_{r['kick']:06d}.jpg"
    cv2.imwrite(str(stills), overlay(clip, r, cameras, r["kick"], ball_rows))
    print(f"wrote {stills}")

    scratch = ROOT / "outputs" / f"_ball_{clip}"
    scratch.mkdir(exist_ok=True)
    for path in scratch.glob("*.jpg"):
        path.unlink()
    written = 0
    for frame in range(r["window"][0] - PAD, r["window"][1] + PAD + 1):
        image = overlay(clip, r, cameras, frame, ball_rows)
        if image is not None:
            cv2.imwrite(str(scratch / f"{written:04d}.jpg"), image)
            written += 1
    video = ROOT / "outputs" / f"ball_fit_{clip}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8",
                    "-i", str(scratch / "%04d.jpg"), "-pix_fmt", "yuv420p", str(video)], check=True)
    for path in scratch.glob("*.jpg"):
        path.unlink()
    scratch.rmdir()
    print(f"wrote {video} ({written} frames at 8 fps, slowed down from {r['fps']})")
    print(f"wrote {shadow_plot(clip, r, ball_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default="SNGS-043")
    parser.add_argument("--frame", type=int, default=None)
    args = parser.parse_args()
    main(args.clip, args.frame)
