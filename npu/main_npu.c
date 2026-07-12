/* main_npu.c — entry point (pure C) for the NPU age/gender detector.
 *
 * Same shape as the CPU build's main.c: parse CLI, install a SIGINT handler,
 * then hand off to the C++/OpenCV+QNN core through the C API in app_npu.h.
 * The heavy lifting (SCRFD face detection + genderage, both on the Hexagon
 * NPU) lives behind this boundary in app_npu.cpp. */
#include "app_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static void on_sigint(int sig) { (void)sig; agn_request_stop(); }

static void usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --device PATH     camera device        (default /dev/video19)\n"
           "  --size WxH        capture size         (default 640x642)\n"
           "  --port N          MJPEG stream port    (default 8092)\n"
           "  --weights DIR     folder with the two model subdirs (default ./weights)\n"
           "  --qairt DIR       QAIRT SDK root        (default ~/qairt/2.42.0.251225)\n"
           "  --conf F          face threshold        (default 0.5)\n"
           "  --every N         detect every N frames (default 1)\n"
           "  --rotate N        rotate 0/90/180/270   (default 180)\n"
           "  -h, --help        this help\n", prog);
}

int main(int argc, char** argv) {
    agn_config cfg;
    cfg.device = "/dev/v4l/by-id/usb-NOVATEK_ASJ_ZNX_NVT_510550000000100-video-index0";
    cfg.width = 640;
    cfg.height = 642;
    cfg.port = 8092;
    cfg.weights_dir = "./weights";
    cfg.qairt_root = NULL;   /* NULL -> app_npu.cpp fills in $HOME/qairt/2.42.0.251225 */
    cfg.conf = 0.5f;
    cfg.detect_every = 1;
    cfg.rotate = 180;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) cfg.device = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            int w, h; if (sscanf(argv[++i], "%dx%d", &w, &h) == 2) { cfg.width = w; cfg.height = h; }
        }
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) cfg.weights_dir = argv[++i];
        else if (!strcmp(argv[i], "--qairt") && i + 1 < argc) cfg.qairt_root = argv[++i];
        else if (!strcmp(argv[i], "--conf") && i + 1 < argc) cfg.conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) cfg.detect_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rotate") && i + 1 < argc) cfg.rotate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (cfg.detect_every < 1) cfg.detect_every = 1;

    setvbuf(stdout, NULL, _IOLBF, 0);   /* live logs even when piped to a file */

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);           /* survive a browser disconnecting mid-stream */

    printf("age/gender (NPU): device=%s size=%dx%d port=%d weights=%s\n",
           cfg.device, cfg.width, cfg.height, cfg.port, cfg.weights_dir);
    return agn_run(&cfg);
}
