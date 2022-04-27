#pragma once

#include <cuda_runtime.h>

#include "math/vector-math.cuh"

// mat2x2

__device__ inline float2 mul(const float2* const m, const float2& v) {
	return {
		m[0].x * v.x + m[1].x * v.y,
		m[0].y * v.x + m[1].y * v.y
	};
}

// mat3x3

__device__ inline float3 mul(const float3* const m, const float3& v) {
	return {
		m[0].x * v.x + m[1].x * v.y + m[2].x * v.z,
		m[0].y * v.x + m[1].y * v.y + m[2].y * v.z,
		m[0].z * v.x + m[1].z * v.y + m[2].z * v.z
	};
}

// mat4x4

__device__ inline float4 mul(const float4* const m, const float4& v) {
	return {
		m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x * v.w,
		m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y * v.w,
		m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z * v.w,
		m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * v.w
	};
}

// special

__device__ inline float3 mulWithTranslation(const float4* const m, const float3& v) {
	float3 result = {
		m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x * 1.0f,
		m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y * 1.0f,
		m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z * 1.0f
	};
	return div(result, m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * 1.0f);
}

__device__ inline float3 mulWithoutTranslation(const float4* const m, const float3& v) {
	float3 result = {
		m[0].x * v.x + m[1].x * v.y + m[2].x * v.z,
		m[0].y * v.x + m[1].y * v.y + m[2].y * v.z,
		m[0].z * v.x + m[1].z * v.y + m[2].z * v.z
	};
	return div(result, m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * 1.0f);
}
