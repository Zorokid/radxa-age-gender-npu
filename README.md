# radxa-age-gender-npu

Real-time **age + gender detection** on the Radxa Dragon Q6A (Qualcomm QCS6490)
using a cheap **HP60C USB camera**, written in C/C++ with OpenCV. Faces are boxed
and labelled (`Male (25-32)`, `Female (8-12)`, …) and the annotated video is served
as an **MJPEG HTTP stream** you watch in any browser on the LAN.

Two variants live in this repo:

- **CPU baseline** (`src/`) — OpenCV DNN on the ARM cores, port 8091
- **NPU offload** (`npu/`) — inference accelerated on the Hexagon NPU via
  Qualcomm **QNN**, port 8092

```
HP60C (USB /dev/video19, MJPG)
   │  v4l2-ctl pipe → JPEG frames → cv::imdecode
   ▼
app.cpp ─ SSD face detect ─ age net + gender net (Levi-Hassner) ─ draw overlay
   │                          (CPU: OpenCV DNN · NPU: QNN runtime)
   ▼
MJPEG HTTP server  →  http://<BOARD_IP>:8091/  (any LAN browser)
```

## Why the capture path is unusual (hard-won)

The HP60C is a non-compliant UVC camera. `cv::VideoCapture` returns black
frames, its only reliable mode is **MJPG 640x642** (720p exceeds USB 2.0
bandwidth and yields 0 bytes), and it emits **spurious `FFD9` end-of-image
bytes mid-frame**. So this project:

1. captures raw bytes by piping `v4l2-ctl --stream-mmap --stream-to=/dev/stdout`,
2. frames the MJPEG on **SOI→next-SOI** boundaries instead of EOI,
3. lets junk blobs fail `imdecode` and drop out naturally.

## Build & run (on the board, Ubuntu 24.04 aarch64)

```bash
cd ~/Radxa_camera_age
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target age_gender -j4
sh run.sh                     # CPU variant → http://<BOARD_IP>:8091/
sh npu/run_npu.sh             # NPU variant → http://<BOARD_IP>:8092/
```

Flags: `--device --size --port --models --conf --every`.
`scripts/selftest.sh` / `scripts/diag.sh` verify headlessly (start detached,
grab one frame via localhost, stop) — useful because streaming under load can
disturb the board's USB WiFi.

## Layout

| Path | What |
|------|------|
| `src/main.c` | C entry point (CLI, signals) |
| `src/app.h` / `src/app.cpp` | C API + OpenCV core (capture, detect, stream) |
| `npu/` | QNN-accelerated variant (`main_npu.c`, `app_npu.cpp`, `qnn/`) |
| `models/` | SSD face detector + Levi-Hassner age & gender models (fetched on board) |
| `run.sh`, `npu/run_npu.sh` | launchers |
| `scripts/` | headless self-test / diagnostics |
| `src/*_test.cpp` | camera / format / still-image diagnostics |
| `HUONG_DAN_CHAY.txt`, `npu/HUONG_DAN_NPU.txt` | step-by-step guides (Vietnamese) |

Gender is fairly accurate; age is a rough bracket (classic Levi-Hassner model —
swappable for a newer ONNX model without touching the pipeline).
