#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "math/matrix-math.cuh"
#include "math/rt-math.cuh"
#include "math/vector-math.cuh"
#include "math/ray-aabb-intersection.cuh"
#include "math/ray-octree-intersection.cuh"

#include "graphics/camera.h"

#include "util/buffer.cuh"

#define XCOORD blockIdx.x * blockDim.x + threadIdx.x
#define YCOORD blockIdx.y * blockDim.y + threadIdx.y

struct Payload {
    unsigned int width, height;
    Camera camera;
};

__global__ void frameToUChar3(Buffer<float3> frame, Buffer<uchar3> result, Buffer<Payload> payloadBuffer) {
    int x = XCOORD;
    int y = YCOORD;
    const Payload payload = payloadBuffer.get();
    
    if (x < payload.width && y < payload.height) {
		int pid = y * payload.width + x;

        float3 color = frame.get(pid);

        result.set(pid, {
            static_cast<unsigned char>(__saturatef(color.x) * 255.0f),
            static_cast<unsigned char>(__saturatef(color.y) * 255.0f),
            static_cast<unsigned char>(__saturatef(color.z) * 255.0f)
        });
    }
}