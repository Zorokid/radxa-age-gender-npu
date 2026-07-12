/* app_npu.h — C API for the NPU age/gender detector.
 *
 * main_npu.c (pure C) fills this config and calls agn_run(); all the
 * OpenCV + QNN/Hexagon-NPU work lives behind this boundary in app_npu.cpp. */
#ifndef AGN_APP_H
#define AGN_APP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* device;      /* e.g. "/dev/video19"                          */
    int         width;       /* capture width  (640 for HP60C MJPG)          */
    int         height;      /* capture height (642 for HP60C MJPG)          */
    int         port;        /* HTTP MJPEG stream port                       */
    const char* weights_dir; /* holds qnn_det10g_test/ and qnn_genderage_test/ */
    const char* qairt_root;  /* QAIRT SDK root; NULL -> $HOME/qairt/2.42.0.251225 */
    float       conf;        /* face detection confidence threshold          */
    int         detect_every;/* run detection every N frames (>=1)           */
    int         rotate;      /* rotate frame: 0/90/180/270 degrees           */
} agn_config;

/* Blocking; returns 0 on clean shutdown (SIGINT), non-zero on fatal error. */
int agn_run(const agn_config* cfg);

/* Ask agn_run() to stop (call from a signal handler). */
void agn_request_stop(void);

#ifdef __cplusplus
}
#endif
#endif
