/* app.h — C API for the real-time age/gender detector.
 *
 * main.c (pure C) fills this config and calls ag_run(); all the OpenCV/C++
 * work lives behind this boundary in app.cpp. */
#ifndef AG_APP_H
#define AG_APP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* device;      /* e.g. "/dev/video19"                      */
    int         width;       /* capture width  (640 for HP60C MJPG)      */
    int         height;      /* capture height (642 for HP60C MJPG)      */
    int         port;        /* HTTP MJPEG stream port (e.g. 8091)       */
    const char* model_dir;   /* folder holding the 6 model files         */
    float       conf;        /* face detection confidence threshold      */
    int         detect_every;/* run age/gender every N frames (>=1)      */
    int         rotate;      /* rotate frame: 0/90/180/270 degrees        */
} ag_config;

/* Blocking; returns 0 on clean shutdown (SIGINT), non-zero on fatal error. */
int ag_run(const ag_config* cfg);

/* Ask ag_run() to stop (call from a signal handler). */
void ag_request_stop(void);

#ifdef __cplusplus
}
#endif
#endif
