// mjpg_test.cpp — try capturing via MJPG (the camera's 30fps modes) through
// OpenCV, which uses libv4l to decode MJPG. Saves one frame.
#include <opencv2/opencv.hpp>
#include <cstdio>
int main(int argc, char** argv) {
    const char* dev = (argc > 1) ? argv[1] : "/dev/video19";
    int W = (argc > 2) ? atoi(argv[2]) : 1280;
    int H = (argc > 3) ? atoi(argv[3]) : 720;
    const char* out = (argc > 4) ? argv[4] : "/tmp/mjpg.png";
    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    if (!cap.isOpened()) { std::fprintf(stderr, "open fail\n"); return 1; }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);
    std::printf("fourcc set, size %.0fx%.0f\n",
                cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Mat f;
    for (int i = 0; i < 15; ++i) { cap.read(f); if (!f.empty()) break; }
    if (f.empty()) { std::fprintf(stderr, "no frame\n"); return 2; }
    cv::Scalar mean, std; cv::meanStdDev(f, mean, std);
    std::printf("frame %dx%d mean=%.1f,%.1f,%.1f std=%.1f,%.1f,%.1f\n",
                f.cols, f.rows, mean[0],mean[1],mean[2], std[0],std[1],std[2]);
    cv::imwrite(out, f);
    std::printf("saved %s\n", out);
    return 0;
}
