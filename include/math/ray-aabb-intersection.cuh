#pragma once

#include <cuda_runtime.h>

#include "math/rt-math.cuh"

__device__ bool intersect(const Ray& ray, const AABB& aabb, float& t) {

	float tMin = 0, tMax = MAX_T;
	float t0, t1;

	// x-axis
	t0 = (aabb.min.x - ray.org.x) * ray.invDir.x;
	t1 = (aabb.max.x - ray.org.x) * ray.invDir.x;

	tMin = max(tMin, min(t0, t1));
	tMax = min(tMax, max(t0, t1));

	if (tMin >= tMax) return false;
	t = tMin;

	// y-axis
	t0 = (aabb.min.y - ray.org.y) * ray.invDir.y;
	t1 = (aabb.max.y - ray.org.y) * ray.invDir.y;

	tMin = max(tMin, min(t0, t1));
	tMax = min(tMax, max(t0, t1));

	if (tMin >= tMax) return false;
	t = min(t, tMin);

	// z-axis
	t0 = (aabb.min.z - ray.org.z) * ray.invDir.z;
	t1 = (aabb.max.z - ray.org.z) * ray.invDir.z;

	tMin = max(tMin, min(t0, t1));
	tMax = min(tMax, max(t0, t1));

	if (tMin >= tMax) return false;
	t = min(t, tMin);

	return true;
}
