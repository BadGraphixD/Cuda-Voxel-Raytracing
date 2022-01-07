#include "util/grid-layout.h"

GridLayout::GridLayout(uint2 blockThreads, const Window& window) {

    threads = dim3(blockThreads.x, blockThreads.y);

    int blocksX = (window.getWidth()  + blockThreads.x - 1) / blockThreads.x;
    int blocksY = (window.getHeight() + blockThreads.y - 1) / blockThreads.y;

    blocks = dim3(blocksX, blocksY);
}