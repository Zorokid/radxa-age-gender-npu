# Radxa_camera_age

Real-time **age + gender detection** on the Radxa Dragon Q6A using the **HP60C USB
camera**, written in C/C++ with OpenCV. Faces are boxed and labelled
(`Male (25-32)`, `Female (8-12)`, …) and the annotated video is served as an
**MJPEG HTTP stream** you watch in a browser.

```
HP60C (USB /dev/video19, MJPG)
   │  v4l2-ctl pipe → JPEG frames → cv::imdecode
   ▼
app.cpp ─ SSD face detect ─ age net + gender net (Levi-Hassner) ─ draw overlay
   │
   ▼
MJPEG HTTP server  →  http://<IP_CUA_BOARD>:8091/  (any browser on the LAN)
```

## Run (on the board)

```bash
ssh radxa@<IP_CUA_BOARD>
cd ~/Radxa_camera_age
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target age_gender -j4
sh run.sh
# open http://<IP_CUA_BOARD>:8091/
```

See **`HUONG_DAN_CHAY.txt`** for full step-by-step instructions (Vietnamese) and
**`CLAUDE.md`** for architecture and the camera gotchas.

## Layout

| Path | What |
|------|------|
| `src/main.c` | C entry point (CLI, signals) |
| `src/app.h` / `src/app.cpp` | C API + OpenCV core (capture, detect, stream) |
| `models/` | SSD face detector + Levi-Hassner age & gender models |
| `run.sh` | start the detector |
| `scripts/` | `selftest.sh`, `diag.sh` — headless verification |
| `src/*_test.cpp` | camera / format / still-image diagnostics |

Inference runs on CPU. Gender is fairly accurate; age is a rough bracket (old
model) — upgradable to a newer ONNX model or the Hexagon NPU (QNN) later.
