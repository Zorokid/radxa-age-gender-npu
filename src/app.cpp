// app.cpp — real-time age/gender detector core (behind the C API in app.h).
//
// Capture:  v4l2-ctl streams MJPG (the HP60C's only working real mode, 640x642)
//           to a pipe; we split JPEG frames and cv::imdecode them. This is far
//           more reliable on this camera than cv::VideoCapture (which returned
//           black/garbage frames here).
// Detect:   OpenCV SSD face detector, then Levi-Hassner age + gender nets.
// Output:   an MJPEG HTTP stream (multipart/x-mixed-replace) so the annotated
//           video can be watched live in any browser on the LAN.

#include "app.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using std::vector;
using std::string;

// ---- shutdown flag -------------------------------------------------------
static std::atomic<bool> g_stop{false};
extern "C" void ag_request_stop(void) { g_stop.store(true); }

// ---- Levi-Hassner constants ---------------------------------------------
static const cv::Scalar MODEL_MEAN(78.4263377603, 87.7689143744, 114.895847746);
static const char* AGE_BUCKETS[] = {
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
};
static const char* GENDERS[] = { "Male", "Female" };

// ---- latest annotated JPEG shared with HTTP clients ----------------------
struct FrameHub {
    std::mutex m;
    std::condition_variable cv;
    vector<uchar> jpeg;
    uint64_t seq = 0;   // bumped each new frame
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
    // Drain the request line(s); we ignore the path and always stream.
    char req[1024];
    recv(fd, req, sizeof(req), 0);

    const char* hdr =
        "HTTP/1.0 200 OK\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    fprintf(stderr, "[http] client connected\n"); fflush(stderr);
    if (!send_all(fd, hdr, strlen(hdr))) { close(fd); return; }

    uint64_t last = 0;
    while (!g_stop.load()) {
        vector<uchar> jpg;
        {
            std::unique_lock<std::mutex> lk(g_hub.m);
            g_hub.cv.wait_for(lk, std::chrono::milliseconds(500),
                              [&]{ return g_hub.seq != last || g_stop.load(); });
            if (g_stop.load()) break;
            if (g_hub.seq == last) continue;   // timed out, no new frame
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

// ================= detection ==============================================
struct Face { cv::Rect box; string label; };

static void run_detection(cv::dnn::Net& face, cv::dnn::Net& age,
                          cv::dnn::Net& gender, const cv::Mat& frame,
                          float conf, vector<Face>& out) {
    out.clear();
    const int W = frame.cols, H = frame.rows;
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300),
                                          cv::Scalar(104, 177, 123), false, false);
    face.setInput(blob);
    cv::Mat det = face.forward();  // [1,1,N,7]
    cv::Mat d(det.size[2], det.size[3], CV_32F, det.ptr<float>());

    for (int i = 0; i < d.rows; ++i) {
        float c = d.at<float>(i, 2);
        if (c < conf) continue;
        int x1 = (int)(d.at<float>(i, 3) * W);
        int y1 = (int)(d.at<float>(i, 4) * H);
        int x2 = (int)(d.at<float>(i, 5) * W);
        int y2 = (int)(d.at<float>(i, 6) * H);
        // pad a little so age/gender see hair/chin context
        int pad = (int)(0.15 * (y2 - y1));
        x1 = std::max(0, x1 - pad); y1 = std::max(0, y1 - pad);
        x2 = std::min(W - 1, x2 + pad); y2 = std::min(H - 1, y2 + pad);
        if (x2 - x1 < 16 || y2 - y1 < 16) continue;

        cv::Rect r(x1, y1, x2 - x1, y2 - y1);
        cv::Mat faceImg = frame(r).clone();
        cv::Mat fblob = cv::dnn::blobFromImage(faceImg, 1.0, cv::Size(227, 227),
                                               MODEL_MEAN, false);
        gender.setInput(fblob);
        cv::Mat gp = gender.forward();
        int gi = (gp.at<float>(0, 0) >= gp.at<float>(0, 1)) ? 0 : 1;

        age.setInput(fblob);
        cv::Mat ap = age.forward();
        cv::Point maxLoc;
        cv::minMaxLoc(ap.reshape(1, 1), nullptr, nullptr, nullptr, &maxLoc);
        int age_idx = maxLoc.x;

        Face f;
        f.box = r;
        f.label = string(GENDERS[gi]) + " " + AGE_BUCKETS[age_idx];
        out.push_back(f);
    }
}

static void draw(cv::Mat& frame, const vector<Face>& faces, double fps) {
    for (const auto& f : faces) {
        cv::rectangle(frame, f.box, cv::Scalar(0, 220, 0), 2);
        int base = 0;
        cv::Size ts = cv::getTextSize(f.label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base);
        cv::Point org(f.box.x, std::max(ts.height + 4, f.box.y - 6));
        cv::rectangle(frame, org + cv::Point(0, base + 2),
                      org + cv::Point(ts.width, -ts.height - 2),
                      cv::Scalar(0, 220, 0), cv::FILLED);
        cv::putText(frame, f.label, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 2);
    }
    char hud[64];
    snprintf(hud, sizeof(hud), "%.1f fps  faces:%zu", fps, faces.size());
    cv::putText(frame, hud, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);
}

// ================= entry ==================================================
extern "C" int ag_run(const ag_config* cfg) {
    string md = cfg->model_dir;
    cv::dnn::Net face, age, gender;
    try {
        face = cv::dnn::readNet(md + "/opencv_face_detector_uint8.pb",
                                md + "/opencv_face_detector.pbtxt");
        age = cv::dnn::readNetFromCaffe(md + "/age_deploy.prototxt",
                                        md + "/age_net.caffemodel");
        gender = cv::dnn::readNetFromCaffe(md + "/gender_deploy.prototxt",
                                           md + "/gender_net.caffemodel");
    } catch (const cv::Exception& e) {
        fprintf(stderr, "model load failed: %s\n", e.what());
        return 2;
    }
    for (auto* n : {&face, &age, &gender}) {
        n->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        n->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    printf("models loaded from %s\n", md.c_str());

    // Start the capture pipe: v4l2-ctl -> concatenated MJPG on stdout.
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "v4l2-ctl -d %s --set-fmt-video=width=%d,height=%d,pixelformat=MJPG "
        "--stream-mmap --stream-count=0 --stream-to=/dev/stdout 2>/dev/null",
        cfg->device, cfg->width, cfg->height);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) { fprintf(stderr, "popen failed\n"); return 3; }

    // HTTP server.
    int lfd = make_listen_socket(cfg->port);
    if (lfd < 0) { fprintf(stderr, "cannot bind port %d\n", cfg->port); pclose(pipe); return 4; }
    std::thread acceptor([lfd]{
        fprintf(stderr, "[http] acceptor listening (fd=%d)\n", lfd); fflush(stderr);
        while (!g_stop.load()) {
            int cfd = accept(lfd, nullptr, nullptr);
            if (cfd < 0) { fprintf(stderr, "[http] accept err\n"); fflush(stderr); break; }
            std::thread(serve_client, cfd).detach();
        }
    });
    printf("MJPEG stream: http://<board-ip>:%d/   (Ctrl+C to stop)\n", cfg->port);

    // Read loop: frame the JPEGs out of the byte stream.
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
        // This camera's MJPG contains spurious FFD9 bytes mid-frame, so we do
        // NOT cut on EOI; we cut when the NEXT frame's SOI appears (imdecode
        // ignores any trailing bytes after the real EOI).
        const size_t NPOS = (size_t)-1;
        for (;;) {
            size_t s0 = NPOS;
            for (size_t i = 0; i + 1 < buf.size(); ++i)
                if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { s0 = i; break; }
            if (s0 == NPOS) { if (buf.size() > 1) buf.clear(); break; }

            size_t s1 = NPOS;
            for (size_t i = s0 + 2; i + 1 < buf.size(); ++i)
                if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { s1 = i; break; }
            if (s1 == NPOS) {   // next frame not here yet
                if (s0 > 0) buf.erase(buf.begin(), buf.begin() + s0);
                break;
            }

            vector<uchar> one(buf.begin() + s0, buf.begin() + s1);
            buf.erase(buf.begin(), buf.begin() + s1);

            // The camera interleaves small junk blobs between real frames;
            // those fail to decode and are simply skipped.
            cv::Mat frame = cv::imdecode(one, cv::IMREAD_COLOR);
            if (frame.empty()) continue;

            // The HP60C is mounted upside down; rotate to upright.
            if (cfg->rotate == 90)  cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            else if (cfg->rotate == 180) cv::rotate(frame, frame, cv::ROTATE_180);
            else if (cfg->rotate == 270) cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);

            if (frameNo % cfg->detect_every == 0)
                run_detection(face, age, gender, frame, cfg->conf, faces);

            // fps (EMA)
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
                fprintf(stderr, "[cap] frame %ld  %dx%d  jpg=%zuB  %.1ffps  faces=%zu\n",
                        frameNo, frame.cols, frame.rows, jpg.size(), fps, faces.size());
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
    printf("\nstopped after %ld frames\n", frameNo);
    return 0;
}
