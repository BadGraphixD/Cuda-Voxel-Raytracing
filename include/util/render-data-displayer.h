#pragma once

#include "graphics/window.h"

class RenderDataDisplayer {
private:
    GLuint texture;
    GLuint pbo;

public:
    RenderDataDisplayer(Window* window);
    void display(void* data, Window* window);
};