#include "HalideRuntime.h"

#define WEAK __attribute__((weak))

extern "C" {

extern void *glXGetProcAddressARB(const char *);
extern void *XOpenDisplay(void *);
extern int XDefaultScreen(void *);
extern int glXQueryExtension(void *, void *, void *);
extern void *glXChooseVisual(void *, int, int *);
extern void *glXCreateContext(void *, void *, int, int);
extern void **glXChooseFBConfig(void *, int, const int *, int *);
extern unsigned long glXCreatePbuffer(void *, void *, const int *);
extern int XFree(void *);
extern int XSync(void *, int);
extern void *glXGetCurrentContext();
extern int glXMakeContextCurrent(void *, unsigned long, unsigned long, void *);

WEAK void *halide_opengl_get_proc_address(void *user_context, const char *name) {
    return glXGetProcAddressARB(name);
}

// Initialize OpenGL
WEAK int halide_opengl_create_context(void *user_context) {
    if (glXGetCurrentContext()) {
        // Already have a context
        return 0;
    }

    void *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        halide_error(user_context, "Could not open X11 display.\n");
        return 1;
    }

    // GLX supported?
    if (!glXQueryExtension(dpy, NULL, NULL)) {
        halide_error(user_context, "GLX not supported by X server.\n");
        return 1;
    }

    int attribs[] = {4 /* GLX_RGBA */, 0 /* Sentinel */};
    int screen = XDefaultScreen(dpy);

    void *vi = glXChooseVisual(dpy, screen, attribs);
    if (!vi) {
        halide_error(user_context, "Could not find suitable visual.\n");
        return 1;
    }

    void *ctx = glXCreateContext(dpy, vi, 0, 1);
    if (!ctx) {
        halide_error(user_context, "Could not create OpenGL context.\n");
        return 1;
    }

    int fb_attribs[] = {0};
    int num_configs = 0;
    void** fb_config = glXChooseFBConfig(dpy, screen, fb_attribs, &num_configs);
    if (!num_configs) {
        halide_error(user_context, "Could not get framebuffer config.\n");
        return 1;
    }

    int pbuffer_attribs[] = {
        0x8041 /* GLX_PBUFFER_WIDTH */,  32,
        0x8040 /* GLX_PBUFFER_HEIGHT */, 32,
        0
    };
    unsigned long pbuffer = glXCreatePbuffer(dpy, fb_config[0], pbuffer_attribs);

    // clean up:
    XFree(fb_config);
    XSync(dpy, 0);

    if (!glXMakeContextCurrent(dpy, pbuffer, pbuffer, ctx)) {
        halide_error(user_context, "Could not make context current.\n");
        return 1;
    }

    return 0;
}

}
