#pragma once

#include "util/common.cuh"
#include "util/buffer.cuh"

__global__ void rayToUChar3(Buffer<Ray> rays, Buffer<uchar3> result, Buffer<Payload> payloadBuffer) {
    EXCLUDE_PIXELS_OUTSIDE_FRAME(
        Ray ray = rays.get(pid);

        result.set(pid, {
            static_cast<unsigned char>(__saturatef(ray.dir.x) * 255.0f),
            static_cast<unsigned char>(__saturatef(ray.dir.y) * 255.0f),
            static_cast<unsigned char>(__saturatef(ray.dir.z) * 255.0f)
        });
    );
}
