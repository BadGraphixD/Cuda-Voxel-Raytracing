#include "util/grid-layout.h"

GridLayout::GridLayout(unsigned int blockThreadsX, unsigned int blockThreadsY, const Window& window) {

    threads = dim3(blockThreadsX, blockThreadsY);

    int blocksX = (window.getWidth()  + blockThreadsX - 1) / blockThreadsX;
    int blocksY = (window.getHeight() + blockThreadsY - 1) / blockThreadsY;

    blocks = dim3(blocksX, blocksY);
}