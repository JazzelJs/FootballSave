"""Click the pitch-model points on one frame and save their pixel positions.

Usage: uv run python src/calib/click_points.py data/frames/clip07/00000.jpg
Keys:  SPACE = mark the current point at the mouse cursor
       ENTER = point not visible, skip it
       BACKSPACE = go back one point
Zoom with the toolbar's magnifier button, go back to the full view with its home button.
Saves annotations/clicks/<clip>_<frame>.json and a preview PNG in outputs/.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pitch_model import POINTS

ROOT = Path(__file__).resolve().parents[2]


def main(frame_path):
    frame_path = Path(frame_path)
    for k in plt.rcParams:
        if k.startswith("keymap."):
            plt.rcParams[k] = []  # no matplotlib shortcuts: every key below is ours

    img = plt.imread(frame_path)
    names = list(POINTS)
    clicks, marks = {}, {}
    i = 0
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.set_axis_off()

    def prompt():
        if i < len(names):
            ax.set_title(f"[{i + 1}/{len(names)}]  {names[i]}    SPACE mark · ENTER skip · BACKSPACE back")
        else:
            ax.set_title("Done: close the window to save (BACKSPACE to redo the last point)")
        fig.canvas.draw_idle()

    def on_key(e):
        nonlocal i
        if e.key == " " and i < len(names) and e.inaxes is ax:
            h, w = img.shape[:2]
            if not (3 <= e.xdata <= w - 4 and 3 <= e.ydata <= h - 4):
                ax.set_title(f"{names[i]}: that's the edge of the picture, not a corner. ENTER = not visible, skip")
                fig.canvas.draw_idle()
                return
            name = names[i]
            clicks[name] = [round(float(e.xdata), 1), round(float(e.ydata), 1)]
            (dot,) = ax.plot(*clicks[name], "r+", ms=14, mew=1.5)
            label = ax.annotate(name, clicks[name], xytext=(6, -6), textcoords="offset points",
                                color="yellow", fontsize=8)
            marks[name] = (dot, label)
            i += 1
        elif e.key == "enter" and i < len(names):
            i += 1
        elif e.key == "backspace" and i > 0:
            i -= 1
            clicks.pop(names[i], None)
            for artist in marks.pop(names[i], ()):
                artist.remove()
        else:
            return
        prompt()

    fig.canvas.mpl_connect("key_press_event", on_key)
    prompt()
    plt.show()

    if i < len(names):
        print(f"Window closed at point {i + 1}/{len(names)}: saving the {len(clicks)} points marked so far.")
    out = ROOT / "annotations" / "clicks" / f"{frame_path.parent.name}_{frame_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frame": str(frame_path), "points_px": clicks}, indent=2) + "\n")

    h, w = img.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f"{frame_path}: {len(clicks)} points")
    preview = ROOT / "outputs" / f"clicks_{frame_path.parent.name}_{frame_path.stem}.png"
    fig.savefig(preview, dpi=100, bbox_inches="tight")
    print(f"Saved {len(clicks)} points -> {out.relative_to(ROOT)}  (preview: {preview.relative_to(ROOT)})")


if __name__ == "__main__":
    main(sys.argv[1])
