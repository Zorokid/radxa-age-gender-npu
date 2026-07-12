// decode_test.cpp — decode one raw packed-422 frame with several interpretations
// so we can see which one yields a correct image for the HP60C.
// Input: a raw file of width*height*2 bytes (default 1280x1040).
// Output: <out>_YUYV.png, _YVYU.png, _UYVY.png

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    const char* in  = (argc > 1) ? argv[1] : "/tmp/hp60c_yuyv.raw";
    int W           = (argc > 2) ? atoi(argv[2]) : 1280;
    int H           = (argc > 3) ? atoi(argv[3]) : 1040;
    const char* pfx = (argc > 4) ? argv[4] : "/tmp/dec";

    size_t need = (size_t)W * H * 2;
    std::ifstream f(in, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", in); return 1; }
    std::vector<unsigned char> buf(need);
    f.read((char*)buf.data(), need);
    if ((size_t)f.gcount() < need) {
        std::fprintf(stderr, "short read: got %ld need %zu\n", (long)f.gcount(), need);
        return 2;
    }

    cv::Mat packed(H, W, CV_8UC2, buf.data());

    // Extract the luma (Y) plane alone: byte 0,2,4,... of the packed 422 stream.
    // If this grayscale image shows a real scene, the problem is only color
    // decoding; if it too is noise, the sensor isn't producing a real picture.
    {
        cv::Mat y(H, W, CV_8UC1);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                y.at<unsigned char>(r, c) = buf[(size_t)(r * W + c) * 2];
        char path[512];
        std::snprintf(path, sizeof(path), "%s_Y.png", pfx);
        cv::imwrite(path, y);
        cv::Scalar mean, std; cv::meanStdDev(y, mean, std);
        std::printf("Y    mean=%.1f std=%.1f -> %s\n", mean[0], std[0], path);
    }
    struct { int code; const char* name; } fmts[] = {
        { cv::COLOR_YUV2BGR_YUYV, "YUYV" },
        { cv::COLOR_YUV2BGR_YVYU, "YVYU" },
        { cv::COLOR_YUV2BGR_UYVY, "UYVY" },
    };
    for (auto& ff : fmts) {
        cv::Mat bgr;
        cv::cvtColor(packed, bgr, ff.code);
        cv::Scalar m = cv::mean(bgr);
        char path[512];
        std::snprintf(path, sizeof(path), "%s_%s.png", pfx, ff.name);
        cv::imwrite(path, bgr);
        std::printf("%-4s mean(BGR)=%.1f,%.1f,%.1f -> %s\n",
                    ff.name, m[0], m[1], m[2], path);
    }
    return 0;
}
