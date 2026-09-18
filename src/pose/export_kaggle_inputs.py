"""Package what a Kaggle pose run needs for a clip, and check it before you upload 91 MB.

Usage: uv run python src/pose/export_kaggle_inputs.py clip04 10 15 17 9
       uv run python src/pose/export_kaggle_inputs.py clip04 10 15 17 9 --with-frames

Writes data/kaggle/<clip>_pose_inputs.zip: the raw tracker JSON (where the per-frame boxes live)
plus a manifest the notebook asserts against, so a stale upload fails loudly instead of quietly
posing the wrong player. `--with-frames` adds the frames too, for a clip whose frames are not
already a Kaggle dataset -- clip04's are (`jazzeljs/clip04-data-football`), so the default
leaves them out and the zip stays under a megabyte.

We do NOT pre-crop the players, and that is deliberate. HMR2's ViTDetDataset takes the full
frame plus a box and does its own crop, and `src/pose/orient.py` then needs the box centre in
FULL-frame pixels for the CLIFF correction (`crop_to_full`) -- a body HMR2 saw in a crop sits in
a camera aimed at that crop's centre, not down the optical axis. Hand it pre-cropped images and
that correction has nothing to work with, and every facing angle comes out wrong.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "track"))
from clips import fps, frame_path, frames_dir  # noqa: E402

DETECTOR = "football-player-detection-v9_botsort"
MIN_FRAMES = 100  # a track with fewer boxes than this in the window is not worth a GPU session


def boxes_per_frame(clip):
    raw = json.loads((ROOT / "data" / "track" / f"{clip}_{DETECTOR}.json").read_text())
    return {f["frame"]: {b["id"]: b["xyxy"] for b in f["boxes"] if b["cls"] != "ball"}
            for f in raw["frames"]}


def report(clip, tracks, window):
    """The numbers that decide whether this run is worth doing, printed before the upload.

    Crop height is the one that matters. Stage 4 measured SNGS-043's shooter at 78 px tall and
    that set the whole stage's ceiling, because pose models want 256x256 and everything smaller
    gets upscaled. Anything here below ~70 px will be worse than what we already have.
    """
    boxes = boxes_per_frame(clip)
    rows = {}
    for track in tracks:
        heights = [b[3] - b[1] for frame in range(window[0], window[1] + 1)
                   for b in [boxes.get(frame, {}).get(track)] if b is not None]
        rows[track] = heights
        if not heights:
            print(f"  track {track}: NOT FOUND in frames {window[0]}-{window[1]}")
            continue
        print(f"  track {track}: {len(heights)} boxes, "
              f"median {np.median(heights):.0f} px tall (min {min(heights):.0f}, "
              f"max {max(heights):.0f})")
    missing = [t for t, h in rows.items() if len(h) < MIN_FRAMES]
    if missing:
        raise SystemExit(f"tracks {missing} have fewer than {MIN_FRAMES} boxes in the window. "
                         "Pick different ids, or widen the window.")
    return rows


def main(clip, tracks, window, with_frames):
    available = sorted(int(p.stem) for p in frames_dir(clip).glob("*.jpg"))
    window = window or (available[0], available[-1])
    print(f"{clip}: {len(available)} frames on disk ({available[0]}-{available[-1]}), "
          f"{fps(clip)} fps, window {window[0]}-{window[1]}")
    heights = report(clip, tracks, window)

    manifest = {"clip": clip, "fps": fps(clip), "window": list(window),
                "targets": {str(t): [t] for t in tracks},
                "frame_pattern": frame_path(clip, 0).name,
                "boxes_per_track": {str(t): len(h) for t, h in heights.items()},
                "median_box_height_px": {str(t): float(np.median(h)) for t, h in heights.items()}}

    out = ROOT / "data" / "kaggle" / f"{clip}_pose_inputs.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    track_json = ROOT / "data" / "track" / f"{clip}_{DETECTOR}.json"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
        bundle.write(track_json, track_json.name)
        if with_frames:
            # Stored, not deflated: jpgs do not compress and zipping them again just burns time.
            for frame in range(window[0], window[1] + 1):
                path = frame_path(clip, frame)
                if path.is_file():
                    bundle.write(path, f"{clip}/{path.name}", zipfile.ZIP_STORED)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("\nKaggle steps:")
    print(f"  1. Upload this zip as a private dataset (or add it to an existing one).")
    if not with_frames:
        known = ("`jazzeljs/clip04-data-football` (= data/clip04_frames.zip)" if clip == "clip04"
                 else "the SoccerNet frames dataset (= data/soccernet_frames.zip, SNGS-028 + SNGS-043)"
                 if clip.startswith("SNGS-") else "this clip's frames dataset")
        print(f"  2. Attach the frames dataset too — {known}."
              f" Rerun with --with-frames if it is not on Kaggle yet.")
    print("  3. Attach your private SMPL input (basicmodel_m_lbs_10_207_0_v1.1.0.pkl) and, if you"
          " saved one, the private `hmr2_cache` dataset.")
    print(f"  4. Import notebooks/kaggle/football3d_{clip}_poses.ipynb, enable Internet + a T4 GPU,"
          " Run All.")
    print(f"  5. Download each pose_<id>.npz, rename it pose_{clip}_<id>.npz (the clip belongs in")
    print(f"     the name -- two clips can both have a track 17) and put it in data/pose/npz/.")
    print(f"  6. Bake vertices, THEN pack the mesh. HMR2 saves pose parameters, not vertices:")
    print(f"     uv run python src/pose/apply_smpl.py \\")
    print(f"       --input data/pose/npz/pose_{clip}_<id>.npz \\")
    print(f"       --output data/pose/npz/pose_{clip}_<id>_vertices.npz")
    print(f"     uv run python src/pose/export_smpl_mesh.py \\")
    print(f"       data/pose/npz/pose_{clip}_<id>_vertices.npz \\")
    print(f"       data/pose/mesh/pose_{clip}_<id>.smpl --clip {clip} --track <id> --smooth-yaw 5")
    print(f"  7. Add {{ clip: '{clip}', trackId: <id>, file: 'pose/mesh/pose_{clip}_<id>.smpl' }}")
    print(f"     to ALL_POSES in src/viewer/index.html.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("tracks", type=int, nargs="+")
    parser.add_argument("--window", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--with-frames", action="store_true")
    args = parser.parse_args()
    main(args.clip, args.tracks, tuple(args.window) if args.window else None, args.with_frames)
