"""Where a clip's frames are and its frame rate: our clips (clip04) vs SoccerNet GSR clips (SNGS-028)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_soccernet(clip):
    return clip.startswith("SNGS-")


def frames_dir(clip):
    return ROOT / "data" / "soccernet" / clip / "img1" if is_soccernet(clip) else ROOT / "data" / "frames" / clip


def frame_path(clip, frame):
    """SoccerNet: 000001.jpg, 6 digits, starts at 1. Ours: 00000.jpg, 5 digits, starts at 0."""
    return frames_dir(clip) / (f"{frame:06d}.jpg" if is_soccernet(clip) else f"{frame:05d}.jpg")


def fps(clip):
    return 25 if is_soccernet(clip) else 49.95  # clip04's rate from Stage 0
