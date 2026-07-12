#!/bin/sh
# diag.sh — full detached diagnostic: start app, probe socket + curl locally,
# capture everything to /tmp/diag.log. Survives SSH drops.
set -u
cd "$(dirname "$0")/.."
exec >/tmp/diag.log 2>&1

pkill -9 -f age_gender 2>/dev/null
pkill -9 -f 'v4l2-ctl.*ZNX' 2>/dev/null
sleep 2

echo "### starting app"
setsid ./build/age_gender --models ./models --port 8091 </dev/null >/tmp/ag.log 2>&1 &
sleep 8

echo "### listening?"
ss -ltnp 2>/dev/null | grep 8091 || echo NOT_LISTENING
echo "### curl -v localhost"
timeout 5 curl -v http://127.0.0.1:8091/ --output /tmp/stream.mjpg 2>/tmp/curl.log
echo "curl_exit=$?"
head -c 400 /tmp/curl.log
echo
ls -l /tmp/stream.mjpg 2>/dev/null || echo "no stream file"
echo "### app log (stderr diagnostics)"
cat /tmp/ag.log

pkill -9 -f age_gender 2>/dev/null
pkill -9 -f 'v4l2-ctl.*ZNX' 2>/dev/null
echo "### DONE"
