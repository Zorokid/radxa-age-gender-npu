#!/bin/sh
# run_npu.sh — launch the NPU age/gender detector.
#
# IMPORTANT: on this board the HTP/fastRPC layer locates the DSP skel
# (libQnnHtpV68Skel.so) via the CURRENT WORKING DIRECTORY, not ADSP_LIBRARY_PATH.
# So we cd into this dir (which holds the V68 skel + stub, shared by both models)
# before launching. Model weights live under ./weights/{qnn_det10g_test,qnn_genderage_test}.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
QAIRT="${QAIRT_ROOT:-$HOME/qairt/2.42.0.251225}"
WEIGHTS="$HERE/weights"

export LD_LIBRARY_PATH="$QAIRT/lib/aarch64-oe-linux-gcc11.2:$QAIRT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH"

cd "$HERE"   # skel is found relative to CWD
# The HP60C USB camera's /dev/videoN number is not stable (venus codec nodes
# grab video19/20 on some boots), so use the fixed by-id symlink.
CAM="${CAM:-/dev/v4l/by-id/usb-NOVATEK_ASJ_ZNX_NVT_510550000000100-video-index0}"
exec "$HERE/build/age_gender_npu" \
  --device "$CAM" --size 640x642 --port 8092 \
  --weights "$WEIGHTS" --qairt "$QAIRT" "$@"
