#pragma once

#include <cuda_runtime.h>

#include "math/matrix_math.cuh"

#define NEAR_CLIP_DIST 0.1f
#define FAR_CLIP_DIST 1000.0f

#define RGB_LUMINANCE float3({ 0.2126, 0.7152, 0.0722 })

struct Ray {
	float3 org, dir;

	__device__ inline float3 at(const float& t) const {
		return add(org, mul(dir, t));
	}
};

struct AABB {
	float3 min, max;

	__host__ __device__ inline float3 center() const {
		return mul(add(min, max), 0.5f);
	}
};

__device__ inline float3 reflect(const float3& v, const float3& n) {
	return sub(v, mul(n, dot(v, n) * 2.0f));
}

__device__ inline float3 refract(const float3& v, const float3& n, const float& ratio, const float& ct) {
	float3 refracted = mul(add(v, mul(n, ct)), ratio);
	float opl = -sqrtf(fabsf(1.0f - sqrLength(refracted)));

	return add(refracted, mul(n, opl));
}

__device__ inline float reflactance(const float& ct, const float& refIdx) {
	float r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - ct), 5);
}

__device__ inline Ray transform(const float4* matrix, const Ray& ray) {
	float3 p0 = mulWithTranslation(matrix, ray.org);
	float3 p1 = mulWithTranslation(matrix, add(ray.org, ray.dir));

	return { p0, sub(p1, p0) };
}
