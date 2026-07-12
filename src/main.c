/* main.c — entry point (pure C). Parses CLI args, installs a SIGINT handler,
 * and hands off to the C++/OpenCV core through the C API in app.h. */
#include "app.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static void on_sigint(int sig) { (void)sig; ag_request_stop(); }

static void usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --device PATH     camera device      (default /dev/video19)\n"
           "  --size WxH        capture size       (default 640x642)\n"
           "  --port N          MJPEG stream port  (default 8091)\n"
           "  --models DIR      model folder       (default ./models)\n"
           "  --conf F          face threshold     (default 0.5)\n"
           "  --every N         infer every N frms (default 2)\n"
           "  --rotate N        rotate 0/90/180/270 (default 180)\n"
           "  -h, --help        this help\n", prog);
}

int main(int argc, char** argv) {
    ag_config cfg;
    cfg.device = "/dev/v4l/by-id/usb-NOVATEK_ASJ_ZNX_NVT_510550000000100-video-index0";
    cfg.width = 640;
    cfg.height = 642;
    cfg.port = 8091;
    cfg.model_dir = "./models";
    cfg.conf = 0.5f;
    cfg.detect_every = 2;
    cfg.rotate = 180;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) cfg.device = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            int w, h; if (sscanf(argv[++i], "%dx%d", &w, &h) == 2) { cfg.width = w; cfg.height = h; }
        }
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) cfg.model_dir = argv[++i];
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
    signal(SIGPIPE, SIG_IGN);   /* don't die when a browser disconnects mid-stream */

    printf("age/gender detector: device=%s size=%dx%d port=%d models=%s\n",
           cfg.device, cfg.width, cfg.height, cfg.port, cfg.model_dir);
    return ag_run(&cfg);
}
