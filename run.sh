#!/bin/sh
# run.sh — start the real-time age/gender detector (foreground).
# Watch the live annotated video in a browser at:  http://<board-ip>:8091/
# Stop with Ctrl+C.
cd "$(dirname "$0")"
# The HP60C USB camera's /dev/videoN number shifts across reboots/replugs
# (venus codec nodes steal low numbers), so use the stable by-id symlink.
CAM="${CAM:-/dev/v4l/by-id/usb-NOVATEK_ASJ_ZNX_NVT_510550000000100-video-index0}"
pkill -f 'v4l2-ctl.*ZNX' 2>/dev/null
exec ./build/age_gender --device "$CAM" --size 640x642 --port 8091 --models ./models "$@"
