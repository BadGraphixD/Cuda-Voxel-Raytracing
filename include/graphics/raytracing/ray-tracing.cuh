#pragma once

#include "util/common.cuh"

__global__ void traceRays(Buffer<Ray> rays, Octree octree, Buffer<Node> nodes, Buffer<Material> materials, Buffer<float3> frame, Buffer<Payload> payloadBuffer) {
    int x = XCOORD;
    int y = YCOORD;
    const Payload payload = payloadBuffer.get();

    if (x < payload.width && y < payload.height) {
		int pid = y * payload.width + x;

        Ray ray = rays.get(pid);

        const int voxel = octree.traverse(ray, nodes.getPtr(), materials.getPtr());

        // __syncthreads();

        float3 color = ray.dir;

        if (voxel != -1) {
            color = materials.get(nodes.get(voxel).materialIdx).color;
        }

        frame.set(pid, color);
    }
}