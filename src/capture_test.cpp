// capture_test.cpp — verify the HP60C (USB UVC) camera end-to-end via OpenCV.
//
// The HP60C only streams YUYV @ 1280x1040 (its MJPG mode is dead), so we must
// force the pixel format explicitly; OpenCV defaults to MJPG and would fail.
// Grabs a few warm-up frames, saves one as PNG, and prints basic stats.
//
// Usage: capture_test [device] [out.png]   (default /dev/video19, capture.png)

#include <opencv2/opencv.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    const char* dev = (argc > 1) ? argv[1] : "/dev/video19";
    const char* out = (argc > 2) ? argv[2] : "capture.png";

    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", dev);
        return 1;
    }

    // Force YUYV 1280x1040 — the only working mode on this camera.
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1040);
    cap.set(cv::CAP_PROP_CONVERT_RGB, 1.0); // let OpenCV give us BGR

    std::printf("opened %s -> %.0fx%.0f\n", dev,
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat frame;
    for (int i = 0; i < 10; ++i) {          // warm-up: first frames can be empty
        if (!cap.read(frame) || frame.empty()) {
            std::fprintf(stderr, "warm-up frame %d empty\n", i);
            continue;
        }
    }
    if (frame.empty()) {
        std::fprintf(stderr, "ERROR: no valid frame captured\n");
        return 2;
    }

    cv::Scalar m = cv::mean(frame);
    std::printf("frame %dx%d ch=%d mean(BGR)=%.1f,%.1f,%.1f\n",
                frame.cols, frame.rows, frame.channels(), m[0], m[1], m[2]);

    if (!cv::imwrite(out, frame)) {
        std::fprintf(stderr, "ERROR: imwrite %s failed\n", out);
        return 3;
    }
    std::printf("saved %s\n", out);
    return 0;
}
