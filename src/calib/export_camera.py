"""Write camera.json for Stage 2: one H per frame (our pitch meters -> pixels), from PnLCalib.

Usage: uv run python src/calib/export_camera.py clip04
Reads data/pnlcalib/pnlcalib_raw_<clip>.json, writes data/camera/<clip>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

from compare_pnl import camera_matrix, camera_to_h
from homography import load_pairs, reprojection_errors

ROOT = Path(__file__).resolve().parents[2]


def main(clip):
    assert clip == "clip04", "pitch_to_soccernet is clip04-only (goal on the left): generalise it first"
    raw = json.load(open(ROOT / "data" / "pnlcalib" / f"pnlcalib_raw_{clip}.json"))

    frames = []
    for r in raw["frames"]:
        if not r["ok"]:
            continue  # PnLCalib found no camera: leave the frame out, Stage 2 skips it
        H = camera_to_h(camera_matrix(r["cam_params"]))  # your function does the real work
        frames.append({"frame": r["frame"], "H": (H / H[2, 2]).tolist(), "pnl_rep_err_px": r["rep_err_px"]})

    out = ROOT / "data" / "camera" / f"{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "clip": clip,
        "image_size": raw["image_size"],
        "H": "pitch meters (x, y, 1) -> pixels (u, v, w); pixels -> meters = inv(H)",
        "source": raw["method"] + ", converted with compare_pnl.pitch_to_soccernet",
        "frames": frames,
    }, indent=1) + "\n")
    print(f"{len(frames)}/{len(raw['frames'])} frames -> {out.relative_to(ROOT)}")

    # Check: read the file back and test it on your clicks (should match compare_pnl's 5.9 px).
    H_by_frame = {f["frame"]: np.array(f["H"]) for f in json.load(open(out))["frames"]}
    errs = [reprojection_errors(H_by_frame[int(p.stem.split("_")[1])], *load_pairs(p)[1:]).mean()
            for p in sorted((ROOT / "annotations" / "clicks").glob(f"{clip}_*.json"))]
    print(f"read back: mean error on {len(errs)} clicked frames = {np.mean(errs):.1f} px")


if __name__ == "__main__":
    main(sys.argv[1])
