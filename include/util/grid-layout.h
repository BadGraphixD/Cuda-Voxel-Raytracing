#pragma once

#include <cuda_runtime.h>

#include "graphics/window.h"

struct GridLayout {
    dim3 threads, blocks;

    GridLayout() = default;
    GridLayout(unsigned int blockThreadsX, unsigned int blockThreadsY, const Window& window);
};