#!/usr/bin/env bash
# Usage: src/extract_frames.sh <input_video> <clip_name> <start> <end>
# Example: src/extract_frames.sh ~/Downloads/match.mp4 clip01 00:42:10 00:42:18
set -euo pipefail

in="$1"; name="$2"; start="$3"; end="$4"
clip="data/clips/$name.mp4"
frames="data/frames/$name"
rm -rf "data/frames/${name:?}" && mkdir -p "$frames"   # no stale frames from a longer earlier run

# Re-encode instead of -c copy: stream copy can only cut at keyframes, so the clip would start early.
ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$start" -to "$end" -i "$in" -c:v libx264 -crf 18 -an "$clip"

# -start_number 0 so file 00000.jpg = frame 0 in tracks.json
ffmpeg -nostdin -hide_banner -loglevel error -y -i "$clip" -q:v 2 -start_number 0 "$frames/%05d.jpg"

ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 "$clip"
echo "frames written: $(ls "$frames" | wc -l | tr -d ' ') -> $frames"
