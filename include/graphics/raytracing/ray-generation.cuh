#pragma once

#include "util/common.cuh"
#include "util/buffer.cuh"

__device__ inline static void createRay(Ray& ray, const float& x, const float& y, const Payload& payload) {
	ray.org = payload.camera.position;
	ray.dir = normalize(sub(add(
		payload.camera.lowerLeftCorner, add(
			mul(payload.camera.horizontal, 1.0f - x / (float)payload.width),
			mul(payload.camera.vertical, 1.0f - y / (float)payload.height)
		)), ray.org
	));
}

__global__ void generateRays(Buffer<Ray> rays, Buffer<Payload> payloadBuffer) {
    EXCLUDE_PIXELS_OUTSIDE_FRAME(
        Ray ray;
        createRay(ray, x, y, payload);
        rays.set(pid, ray);
    );
}