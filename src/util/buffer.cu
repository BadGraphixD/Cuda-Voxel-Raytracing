#include "util/buffer.cuh"

void BufferManager::destroyBuffers() {
    for (void* buffer : buffers)
        ((Buffer<int>*)buffer)->destroy();
}
