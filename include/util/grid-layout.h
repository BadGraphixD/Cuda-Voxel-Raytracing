#pragma once

#include <cuda_runtime.h>

#include "graphics/window.h"

struct GridLayout {
    dim3 threads, blocks;

    GridLayout() = default;
    GridLayout(uint2 blockThreads, const Window& window);
};