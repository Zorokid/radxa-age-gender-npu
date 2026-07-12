// image_test.cpp — run the face + age + gender pipeline on a still image and
// write an annotated copy. Proves the detection/labeling path independently of
// the live camera.  Usage: image_test <models_dir> <in.jpg> <out.jpg>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdio>
#include <string>

static const cv::Scalar MODEL_MEAN(78.4263377603, 87.7689143744, 114.895847746);
static const char* AGE[] = {"(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"};
static const char* GEN[] = {"Male","Female"};

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s models in out\n", argv[0]); return 1; }
    std::string md = argv[1];
    cv::Mat img = cv::imread(argv[2]);
    if (img.empty()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }

    cv::dnn::Net face = cv::dnn::readNet(md+"/opencv_face_detector_uint8.pb", md+"/opencv_face_detector.pbtxt");
    cv::dnn::Net age  = cv::dnn::readNetFromCaffe(md+"/age_deploy.prototxt", md+"/age_net.caffemodel");
    cv::dnn::Net gen  = cv::dnn::readNetFromCaffe(md+"/gender_deploy.prototxt", md+"/gender_net.caffemodel");

    int W = img.cols, H = img.rows;
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(300,300), cv::Scalar(104,177,123), false, false);
    face.setInput(blob);
    cv::Mat det = face.forward();
    cv::Mat d(det.size[2], det.size[3], CV_32F, det.ptr<float>());

    int n = 0;
    for (int i = 0; i < d.rows; ++i) {
        float c = d.at<float>(i,2);
        if (c < 0.6f) continue;
        int x1=(int)(d.at<float>(i,3)*W), y1=(int)(d.at<float>(i,4)*H);
        int x2=(int)(d.at<float>(i,5)*W), y2=(int)(d.at<float>(i,6)*H);
        int pad=(int)(0.15*(y2-y1));
        x1=std::max(0,x1-pad); y1=std::max(0,y1-pad);
        x2=std::min(W-1,x2+pad); y2=std::min(H-1,y2+pad);
        if (x2-x1<20||y2-y1<20) continue;
        cv::Rect r(x1,y1,x2-x1,y2-y1);
        cv::Mat fb = cv::dnn::blobFromImage(img(r).clone(), 1.0, cv::Size(227,227), MODEL_MEAN, false);
        gen.setInput(fb); cv::Mat gp = gen.forward();
        int gi = gp.at<float>(0,0) >= gp.at<float>(0,1) ? 0 : 1;
        age.setInput(fb); cv::Mat ap = age.forward();
        cv::Point ml; cv::minMaxLoc(ap.reshape(1,1), nullptr,nullptr,nullptr,&ml);
        std::string label = std::string(GEN[gi]) + " " + AGE[ml.x];
        printf("face %d: conf=%.2f box=[%d,%d,%d,%d] -> %s\n", n, c, x1,y1,x2,y2, label.c_str());
        cv::rectangle(img, r, cv::Scalar(0,220,0), 2);
        cv::putText(img, label, cv::Point(x1, std::max(20,y1-6)), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
        n++;
    }
    printf("total faces: %d\n", n);
    cv::imwrite(argv[3], img);
    return 0;
}
