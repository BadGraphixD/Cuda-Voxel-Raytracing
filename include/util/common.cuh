#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "math/matrix_math.cuh"
#include "math/rt_math.cuh"
#include "math/vector_math.cuh"

#include "graphics/camera.h"

#define XCOORD blockIdx.x * blockDim.x + threadIdx.x
#define YCOORD blockIdx.y * blockDim.y + threadIdx.y

#define EXCLUDE_PIXELS_OUTSIDE_FRAME(execute) \
    int x = XCOORD;\
    int y = YCOORD;\
    const Payload payload = payloadBuffer.get();\
    if (x < payload.width && y < payload.height) {\
		int pid = y * payload.width + x;\
        execute\
    }

struct Payload {
    unsigned int width, height;
    Camera camera;
};