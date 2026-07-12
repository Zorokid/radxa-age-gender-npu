#!/bin/sh
# selftest.sh — start the detector in the background, let the camera warm up,
# grab one annotated frame from the MJPEG stream, then stop everything.
# Leaves the frame at /tmp/ag_frame.jpg and prints the app log.
set -u
cd "$(dirname "$0")/.."

# Thorough cleanup of any orphaned instance holding the camera/port.
pkill -f age_gender 2>/dev/null
pkill -f 'v4l2-ctl.*ZNX' 2>/dev/null
sleep 1
pkill -9 -f age_gender 2>/dev/null
pkill -9 -f 'v4l2-ctl.*ZNX' 2>/dev/null
sleep 1

setsid ./build/age_gender --models ./models --port 8091 </dev/null >/tmp/ag.log 2>&1 &
sleep 12

echo "===== APP LOG ====="
cat /tmp/ag.log
echo "===== ALIVE? ====="
if pgrep -f age_gender >/dev/null; then echo "app running"; else echo "APP EXITED EARLY"; fi

echo "===== GRAB FRAME ====="
timeout 6 curl -s http://127.0.0.1:8091/ --output /tmp/stream.mjpg
if [ -f /tmp/stream.mjpg ]; then
  python3 - <<'PY'
d = open("/tmp/stream.mjpg", "rb").read()
print("stream bytes:", len(d))
s = d.find(b"\xff\xd8")
e = d.find(b"\xff\xd9", s + 2) if s >= 0 else -1
if s >= 0 and e >= 0:
    open("/tmp/ag_frame.jpg", "wb").write(d[s:e + 2])
    print("saved /tmp/ag_frame.jpg", e + 2 - s, "bytes")
else:
    print("no complete JPEG in stream")
PY
else
  echo "curl produced no data (server not serving)"
fi

pkill -f age_gender 2>/dev/null
pkill -f 'v4l2-ctl.*ZNX' 2>/dev/null
sleep 1
pkill -9 -f age_gender 2>/dev/null
pkill -9 -f 'v4l2-ctl.*ZNX' 2>/dev/null
echo "===== DONE ====="
