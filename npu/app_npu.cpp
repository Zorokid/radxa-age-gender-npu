// app_npu.cpp — real-time age/gender detector core, Hexagon-NPU edition.
//
// Pipeline:
//   Capture : v4l2-ctl streams MJPG (HP60C's only working mode, 640x642) to a
//             pipe; we frame the JPEGs on SOI boundaries and cv::imdecode them
//             (same trick as the CPU build — cv::VideoCapture is black here).
//   Detect  : InsightFace SCRFD det_10g on the NPU -> faces + 5 landmarks.
//   Age/sex : InsightFace genderage on the NPU per face -> gender + real age.
//   Output  : an MJPEG HTTP stream (multipart/x-mixed-replace) watchable in a
//             browser on the LAN.
//
// Both models run on the Hexagon NPU through qnn_py::QnnRuntime, a thin C++
// wrapper over the QNN SampleApp that loads a prebuilt QCS6490 context .bin and
// executes it on the HTP backend. Inference is float-in / float-out; the .bin
// handles quantisation internally, so we feed the exact NCHW blobs the original
// ONNX models expect.

#include "app_npu.h"

#include <opencv2/opencv.hpp>

#include "qnn/QnnRuntime.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using std::vector;
using std::string;
using qnn_py::QnnRuntime;
using qnn_py::TensorInfo;

// ---- dequant (QNN scale/offset convention) --------------------------------
// real = scale * (quantized + offset)   (offset is the negated zero-point,
// e.g. -128 for these uint8 tensors). Verified against the qnn-net-run
// reference outputs for both det_10g and genderage.
static void dequantize_u8(const uint8_t* q, size_t n, double scale, int32_t offset,
                          vector<float>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = (float)(((double)(int)q[i] + offset) * scale);
}

// ---- shutdown flag -------------------------------------------------------
static std::atomic<bool> g_stop{false};
extern "C" void agn_request_stop(void) { g_stop.store(true); }

// ---- genderage output --------------------------------------------------
static const char* GENDERS[] = { "F", "M" };   // genderage: argmax(fc1[0:2]) -> 0=F, 1=M

// ================= latest annotated JPEG shared with HTTP clients ==========
struct FrameHub {
    std::mutex m;
    std::condition_variable cv;
    vector<uchar> jpeg;
    uint64_t seq = 0;
} g_hub;

static void publish(const vector<uchar>& jpg) {
    std::lock_guard<std::mutex> lk(g_hub.m);
    g_hub.jpeg = jpg;
    g_hub.seq++;
    g_hub.cv.notify_all();
}

// ================= HTTP MJPEG server ======================================
static int make_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (sockaddr*)&a, sizeof(a)) < 0) { close(fd); return -1; }
    if (listen(fd, 8) < 0) { close(fd); return -1; }
    return fd;
}

static bool send_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k; n -= (size_t)k;
    }
    return true;
}

static void serve_client(int fd) {
    char req[1024];
    recv(fd, req, sizeof(req), 0);   // drain request; we ignore the path

    const char* hdr =
        "HTTP/1.0 200 OK\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!send_all(fd, hdr, strlen(hdr))) { close(fd); return; }

    uint64_t last = 0;
    while (!g_stop.load()) {
        vector<uchar> jpg;
        {
            std::unique_lock<std::mutex> lk(g_hub.m);
            g_hub.cv.wait_for(lk, std::chrono::milliseconds(500),
                              [&]{ return g_hub.seq != last || g_stop.load(); });
            if (g_stop.load()) break;
            if (g_hub.seq == last) continue;
            last = g_hub.seq;
            jpg = g_hub.jpeg;
        }
        char part[128];
        int n = snprintf(part, sizeof(part),
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
            jpg.size());
        if (!send_all(fd, part, (size_t)n)) break;
        if (!send_all(fd, (const char*)jpg.data(), jpg.size())) break;
        if (!send_all(fd, "\r\n", 2)) break;
    }
    close(fd);
}

// ================= SCRFD face detection (on NPU) ==========================
// det_10g: input "input_1" [1,3,640,640] f32; 9 outputs, three FPN strides
// {8,16,32}, each with (score[N,1], bbox[N,4], kps[N,10]), N = (640/s)^2 * 2.
// We identify each output tensor by its dims (rows -> stride, cols -> kind)
// instead of by position, because the QNN output order is grouped per-stride.
static const int   SCRFD_SIZE = 640;
static const float SCRFD_MEAN = 127.5f;
static const float SCRFD_STD  = 128.0f;
static const float SCRFD_IOU  = 0.4f;

struct Face {
    cv::Rect2f box;          // in captured-frame pixels
    float      score;
    cv::Point2f kps[5];
    // filled by genderage:
    int   gender = -1;
    int   age    = -1;
};

// Anchor centres for one stride, cached: order is (y,x,anchor) row-major,
// matching numpy mgrid[:h,:w][::-1] stacked twice for the 2 anchors.
static const float* anchor_centers(int stride) {
    static std::vector<float> cache[3];   // strides 8,16,32 -> idx 0,1,2
    int idx = (stride == 8) ? 0 : (stride == 16) ? 1 : 2;
    if (!cache[idx].empty()) return cache[idx].data();
    int hw = SCRFD_SIZE / stride;
    std::vector<float>& c = cache[idx];
    c.resize((size_t)hw * hw * 2 * 2);    // *2 anchors, *2 coords
    size_t p = 0;
    for (int y = 0; y < hw; ++y)
        for (int x = 0; x < hw; ++x)
            for (int a = 0; a < 2; ++a) {
                c[p++] = (float)(x * stride);
                c[p++] = (float)(y * stride);
            }
    return c.data();
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.f, x2 - x1), h = std::max(0.f, y2 - y1);
    float inter = w * h;
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0 ? inter / uni : 0.f;
}

// Map each output blob to (stride, kind) via its TensorInfo dims.
struct OutMap { int score = -1, bbox = -1, kps = -1; };   // indices into outputs

static void build_outmap(const vector<qnn_py::TensorInfo>& info, OutMap m[3]) {
    for (int i = 0; i < (int)info.size(); ++i) {
        if (info[i].dims.empty()) continue;
        uint32_t rows = info[i].dims[0];
        uint32_t cols = info[i].dims.size() > 1 ? info[i].dims[1] : 1;
        int s = (rows == 12800) ? 0 : (rows == 3200) ? 1 : (rows == 800) ? 2 : -1;
        if (s < 0) continue;
        if (cols == 1)      m[s].score = i;
        else if (cols == 4) m[s].bbox  = i;
        else if (cols == 10) m[s].kps  = i;
    }
}

static void scrfd_detect(QnnRuntime& det, const OutMap outmap[3],
                         const TensorInfo& in_info, const vector<TensorInfo>& out_info,
                         const cv::Mat& frame, float conf, vector<Face>& out) {
    out.clear();
    const int W = frame.cols, H = frame.rows;

    // Letterbox into 640x640 top-left, preserving aspect ratio (rest = 0).
    float im_ratio = (float)H / (float)W;
    float model_ratio = 1.0f;   // SCRFD_SIZE / SCRFD_SIZE
    int new_w, new_h;
    if (im_ratio > model_ratio) { new_h = SCRFD_SIZE; new_w = (int)(new_h / im_ratio); }
    else                        { new_w = SCRFD_SIZE; new_h = (int)(new_w * im_ratio); }
    float det_scale = (float)new_h / (float)H;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    cv::Mat canvas = cv::Mat::zeros(SCRFD_SIZE, SCRFD_SIZE, frame.type());
    resized.copyTo(canvas(cv::Rect(0, 0, new_w, new_h)));

    // The det_10g context binary takes its native UFIXED_POINT_8 input as the
    // RAW letterboxed image pixels (0-255), NCHW, RGB (the (x-127.5)/128
    // normalisation is baked into the quantised graph). So feed raw pixels — do
    // NOT normalise or requantise here.
    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0,
                                          cv::Size(SCRFD_SIZE, SCRFD_SIZE),
                                          cv::Scalar(0, 0, 0),
                                          /*swapRB=*/true, /*crop=*/false);
    (void)in_info;
    vector<uint8_t> qin(blob.total());
    { const float* bp = blob.ptr<float>();
      for (size_t i = 0; i < qin.size(); ++i)
          qin[i] = (uint8_t)std::min(255.f, std::max(0.f, std::round(bp[i]))); }
    vector<const uint8_t*> in_ptrs{ qin.data() };
    vector<size_t> in_sizes{ qin.size() };
    vector<vector<uint8_t>> outs = det.runRawPtrs(in_ptrs, in_sizes, 0);

    // Dequantize the 9 native-u8 outputs back to float.
    vector<vector<float>> fout(outs.size());
    for (size_t i = 0; i < outs.size(); ++i)
        dequantize_u8(outs[i].data(), outs[i].size(),
                      out_info[i].scale, out_info[i].offset, fout[i]);

    const int strides[3] = {8, 16, 32};
    vector<Face> cand;
    for (int si = 0; si < 3; ++si) {
        int stride = strides[si];
        const OutMap& om = outmap[si];
        if (om.score < 0 || om.bbox < 0 || om.kps < 0) continue;
        const float* sc  = fout[om.score].data();
        const float* bb  = fout[om.bbox].data();
        const float* kp  = fout[om.kps].data();
        const float* ac  = anchor_centers(stride);
        int hw = SCRFD_SIZE / stride;
        int n = hw * hw * 2;
        for (int i = 0; i < n; ++i) {
            if (sc[i] < conf) continue;
            float cx = ac[i * 2], cy = ac[i * 2 + 1];
            float l = bb[i * 4 + 0] * stride, t = bb[i * 4 + 1] * stride;
            float r = bb[i * 4 + 2] * stride, d = bb[i * 4 + 3] * stride;
            Face f;
            float x1 = (cx - l) / det_scale, y1 = (cy - t) / det_scale;
            float x2 = (cx + r) / det_scale, y2 = (cy + d) / det_scale;
            f.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            f.score = sc[i];
            for (int k = 0; k < 5; ++k) {
                float px = (cx + kp[i * 10 + k * 2 + 0] * stride) / det_scale;
                float py = (cy + kp[i * 10 + k * 2 + 1] * stride) / det_scale;
                f.kps[k] = cv::Point2f(px, py);
            }
            cand.push_back(f);
        }
    }

    // NMS (greedy, score-desc, IoU 0.4).
    std::sort(cand.begin(), cand.end(),
              [](const Face& a, const Face& b){ return a.score > b.score; });
    vector<char> removed(cand.size(), 0);
    for (size_t i = 0; i < cand.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!removed[j] && iou(cand[i].box, cand[j].box) > SCRFD_IOU)
                removed[j] = 1;
    }
}

// ================= genderage (on NPU) =====================================
// genderage: input "data" [1,3,96,96] f32 (RGB, no mean/std); output "fc1"
// [1,3] -> gender = argmax(fc1[0:2]), age = round(fc1[2]*100).
static const int GA_SIZE = 96;

static void genderage_infer(QnnRuntime& ga, const TensorInfo& in_info,
                            const TensorInfo& out_info, const cv::Mat& frame, Face& f) {
    // Similarity-transform crop centred on the face, matching InsightFace's
    // Attribute.preprocess: scale = 96 / (max(w,h) * 1.5).
    float w = f.box.width, h = f.box.height;
    float cx = f.box.x + w * 0.5f, cy = f.box.y + h * 0.5f;
    float scale = (float)GA_SIZE / (std::max(w, h) * 1.5f);
    cv::Matx23f M(scale, 0.f, GA_SIZE * 0.5f - cx * scale,
                  0.f, scale, GA_SIZE * 0.5f - cy * scale);
    cv::Mat aligned;
    cv::warpAffine(frame, aligned, M, cv::Size(GA_SIZE, GA_SIZE),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // genderage native input is also raw pixels (0-255, NCHW, RGB).
    cv::Mat blob = cv::dnn::blobFromImage(aligned, 1.0,
                                          cv::Size(GA_SIZE, GA_SIZE),
                                          cv::Scalar(0, 0, 0),
                                          /*swapRB=*/true, /*crop=*/false);
    (void)in_info;
    vector<uint8_t> qin(blob.total());
    { const float* bp = blob.ptr<float>();
      for (size_t i = 0; i < qin.size(); ++i)
          qin[i] = (uint8_t)std::min(255.f, std::max(0.f, std::round(bp[i]))); }
    vector<const uint8_t*> in_ptrs{ qin.data() };
    vector<size_t> in_sizes{ qin.size() };
    vector<vector<uint8_t>> outs = ga.runRawPtrs(in_ptrs, in_sizes, 0);
    if (outs.empty() || outs[0].size() < 3) return;
    vector<float> p;
    dequantize_u8(outs[0].data(), outs[0].size(), out_info.scale, out_info.offset, p);
    f.gender = (p[0] >= p[1]) ? 0 : 1;
    f.age    = (int)std::lround(p[2] * 100.0f);
}

// ================= overlay ================================================
static void draw(cv::Mat& frame, const vector<Face>& faces, double fps) {
    for (const auto& f : faces) {
        cv::Rect r((int)f.box.x, (int)f.box.y, (int)f.box.width, (int)f.box.height);
        cv::rectangle(frame, r, cv::Scalar(0, 220, 0), 2);
        char lbl[64];
        if (f.gender >= 0) snprintf(lbl, sizeof(lbl), "%s %d", GENDERS[f.gender], f.age);
        else               snprintf(lbl, sizeof(lbl), "%.2f", f.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base);
        cv::Point org(r.x, std::max(ts.height + 4, r.y - 6));
        cv::rectangle(frame, org + cv::Point(0, base + 2),
                      org + cv::Point(ts.width, -ts.height - 2),
                      cv::Scalar(0, 220, 0), cv::FILLED);
        cv::putText(frame, lbl, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 2);
        for (int k = 0; k < 5; ++k)
            cv::circle(frame, f.kps[k], 2, cv::Scalar(0, 160, 255), -1);
    }
    char hud[80];
    snprintf(hud, sizeof(hud), "%.1f fps  faces:%zu  NPU", fps, faces.size());
    cv::putText(frame, hud, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);
}

// ================= helpers ================================================
static string join(const string& a, const string& b) {
    if (!a.empty() && a.back() == '/') return a + b;
    return a + "/" + b;
}

// ================= entry ==================================================
extern "C" int agn_run(const agn_config* cfg) {
    // Resolve paths.
    string qairt = cfg->qairt_root ? cfg->qairt_root : string();
    if (qairt.empty()) {
        const char* home = getenv("HOME");
        qairt = string(home ? home : "/home/radxa") + "/qairt/2.42.0.251225";
    }
    string syslib = join(qairt, "lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so");
    string wdir   = cfg->weights_dir;
    string det_dir = join(wdir, "qnn_det10g_test");
    string ga_dir  = join(wdir, "qnn_genderage_test");

    // Build the two NPU runtimes. Both MUST use the SAME backend .so path so
    // dlopen shares a single HTP backend + device; two different paths load two
    // backend instances and the second QnnDevice_create fails (skel error 1009).
    string backend = join(det_dir, "libQnnHtp.so");
    QnnRuntime det(backend, syslib, join(det_dir, "det_10g_qcs6490.bin"));
    QnnRuntime ga(backend, syslib, join(ga_dir, "genderage_qcs6490.bin"));
    try {
        det.init();
        ga.init();
    } catch (const std::exception& e) {
        fprintf(stderr, "QNN init failed: %s\n", e.what());
        return 2;
    }

    vector<TensorInfo> det_in = det.inputInfo(0);
    vector<TensorInfo> det_out = det.outputInfo(0);
    vector<TensorInfo> ga_in = ga.inputInfo(0);
    vector<TensorInfo> ga_out = ga.outputInfo(0);
    OutMap outmap[3];
    build_outmap(det_out, outmap);
    for (int s = 0; s < 3; ++s)
        if (outmap[s].score < 0 || outmap[s].bbox < 0 || outmap[s].kps < 0) {
            fprintf(stderr, "unexpected SCRFD outputs; cannot map stride %d\n",
                    (s == 0 ? 8 : s == 1 ? 16 : 32));
            return 2;
        }
    printf("NPU models loaded (SCRFD det_10g + genderage) from %s\n", wdir.c_str());

    // Capture pipe: v4l2-ctl -> concatenated MJPG on stdout.
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "v4l2-ctl -d %s --set-fmt-video=width=%d,height=%d,pixelformat=MJPG "
        "--stream-mmap --stream-count=0 --stream-to=/dev/stdout 2>/dev/null",
        cfg->device, cfg->width, cfg->height);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) { fprintf(stderr, "popen failed\n"); return 3; }

    int lfd = make_listen_socket(cfg->port);
    if (lfd < 0) { fprintf(stderr, "cannot bind port %d\n", cfg->port); pclose(pipe); return 4; }
    std::thread acceptor([lfd]{
        while (!g_stop.load()) {
            int cfd = accept(lfd, nullptr, nullptr);
            if (cfd < 0) break;
            std::thread(serve_client, cfd).detach();
        }
    });
    printf("MJPEG stream: http://<board-ip>:%d/   (Ctrl+C to stop)\n", cfg->port);

    vector<uchar> buf;
    buf.reserve(1 << 20);
    uchar chunk[65536];
    vector<Face> faces;
    long frameNo = 0;
    auto t0 = std::chrono::steady_clock::now();
    double fps = 0;

    while (!g_stop.load()) {
        size_t got = fread(chunk, 1, sizeof(chunk), pipe);
        if (got == 0) { if (feof(pipe)) break; else continue; }
        buf.insert(buf.end(), chunk, chunk + got);

        // Frame on SOI boundaries: a complete frame is [SOI_i, SOI_{i+1}).
        const size_t NPOS = (size_t)-1;
        for (;;) {
            size_t s0 = NPOS;
            for (size_t i = 0; i + 1 < buf.size(); ++i)
                if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { s0 = i; break; }
            if (s0 == NPOS) { if (buf.size() > 1) buf.clear(); break; }

            size_t s1 = NPOS;
            for (size_t i = s0 + 2; i + 1 < buf.size(); ++i)
                if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { s1 = i; break; }
            if (s1 == NPOS) {
                if (s0 > 0) buf.erase(buf.begin(), buf.begin() + s0);
                break;
            }

            vector<uchar> one(buf.begin() + s0, buf.begin() + s1);
            buf.erase(buf.begin(), buf.begin() + s1);

            cv::Mat frame = cv::imdecode(one, cv::IMREAD_COLOR);
            if (frame.empty()) continue;

            if (cfg->rotate == 90)  cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            else if (cfg->rotate == 180) cv::rotate(frame, frame, cv::ROTATE_180);
            else if (cfg->rotate == 270) cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);

            if (frameNo % cfg->detect_every == 0) {
                scrfd_detect(det, outmap, det_in[0], det_out, frame, cfg->conf, faces);
                for (auto& f : faces) genderage_infer(ga, ga_in[0], ga_out[0], frame, f);
            }

            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t0).count();
            t0 = now;
            if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

            draw(frame, faces, fps);

            vector<uchar> jpg;
            cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
            publish(jpg);
            frameNo++;
            if (frameNo % 30 == 1) {
                fprintf(stderr, "[cap] frame %ld  %dx%d  %.1ffps  faces=%zu\n",
                        frameNo, frame.cols, frame.rows, fps, faces.size());
                fflush(stderr);
            }
        }
    }

    g_stop.store(true);
    shutdown(lfd, SHUT_RDWR);
    close(lfd);
    g_hub.cv.notify_all();
    if (acceptor.joinable()) acceptor.join();
    pclose(pipe);
    try { det.close(); ga.close(); } catch (...) {}
    printf("\nstopped after %ld frames\n", frameNo);
    return 0;
}
