#include "util/render-data-displayer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "util/log.h"

RenderDataDisplayer::RenderDataDisplayer(Window* window) {

    if (glewInit() != GLEW_OK) {
        Log::Error("Failed to initialize GLEW!");
        return;
    }

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    cudaGLSetGLDevice(0);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window->getWidth(), window->getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    size_t size = window->getWidth() * window->getHeight() * 3;

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void RenderDataDisplayer::display(void* data, Window* window) {

    if (data == nullptr) {
        Log::Error("Pixel data can't be null!");
        return;
    }

    void* dptr = nullptr;
    cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaMemcpy2D(dptr, window->getWidth() * 3, data, window->getWidth() * 3, window->getWidth() * 3, window->getHeight(), cudaMemcpyDeviceToDevice);
    cudaGLUnmapBufferObject(pbo);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window->getWidth(), window->getHeight(), GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 0.0f);
    glEnd();
    
}
