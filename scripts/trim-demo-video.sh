#!/usr/bin/env sh

# trim-demo-video.sh

set -e

VIDEO_FILEPATH="${1:-TRIMMED-pfun-qt6-demo--2026-04-16.mp4}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FPATH="${2:-App_Demo_Showcase--${TIMESTAMP}.mp4}"

ffmpeg -i "$VIDEO_FILEPATH" -filter_complex \
       "[0:v]trim=0:17,setpts=PTS-STARTPTS[v1]; \
 [0:v]trim=17:106,setpts=1/8*(PTS-STARTPTS)[v2]; \
 [0:v]trim=106:147,setpts=PTS-STARTPTS[v3]; \
 [v1][v2][v3]concat=n=3:v=1:a=0[outv]" \
       -map "[outv]" -c:v libx264 -preset fast -crf 22 \
       "$OUTPUT_FPATH"

