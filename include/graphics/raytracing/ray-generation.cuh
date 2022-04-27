#pragma once

#include "util/common.cuh"

__device__ inline static void createRay(Ray& ray, const float& x, const float& y, const Payload& payload) {
	ray.org = payload.camera.position;
	ray.dir = normalize(sub(add(
		payload.camera.lowerLeftCorner, add(
			mul(payload.camera.horizontal, 1.0f - x / (float)payload.width),
			mul(payload.camera.vertical, 1.0f - y / (float)payload.height)
		)), ray.org
	));
	ray.calcInvDir();
}

__global__ void generateRays(Buffer<Ray> rays, Buffer<Payload> payloadBuffer) {
	int x = XCOORD;
    int y = YCOORD;
    const Payload payload = payloadBuffer.get();

    if (x < payload.width && y < payload.height) {
		int pid = y * payload.width + x;
        Ray ray;
        createRay(ray, x, y, payload);
        rays.set(pid, ray);
    }
}