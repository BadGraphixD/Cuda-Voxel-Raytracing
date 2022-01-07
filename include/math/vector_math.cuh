#pragma once

#include <cuda_runtime.h>

#define PI 3.141592653589793f
#define EPSILON 0.000001f
#define MIN_SURFACE_DIST 0.001f
#define MAX_T 1000000.0f
#define AABB_PADDING 0.001f
#define TO_RADIANS (PI / 180.0f)
#define TO_DEGREES (180.0f / PI)

__host__ __device__ inline float3 directionFromAngles(float pitch, float yaw) {
	pitch *= TO_RADIANS;
	yaw *= TO_RADIANS;

	float xzLength = cosf(pitch);
	return { xzLength * cosf(yaw), sinf(pitch), xzLength * sinf(-yaw) };
}

// float2

__host__ __device__ inline float2 add(const float2& v0, const float2& v1) {
	return { v0.x + v1.x, v0.y + v1.y };
}
__host__ __device__ inline float2 add(const float2& v0, const float2& v1, const float2& v2) {
	return { v0.x + v1.x + v2.x, v0.y + v1.y + v2.y };
}
__host__ __device__ inline float2 sub(const float2& v0, const float2& v1) {
	return { v0.x - v1.x, v0.y - v1.y };
}
__host__ __device__ inline float2 mul(const float2& v0, const float2& v1) {
	return { v0.x * v1.x, v0.y * v1.y };
}
__host__ __device__ inline float2 div(const float2& v0, const float2& v1) {
	return { v0.x / v1.x, v0.y / v1.y };
}

__host__ __device__ inline float2 add(const float2& v, const float& f) {
	return { v.x + f, v.y + f };
}
__host__ __device__ inline float2 sub(const float2& v, const float& f) {
	return { v.x - f, v.y - f };
}
__host__ __device__ inline float2 mul(const float2& v, const float& f) {
	return { v.x * f, v.y * f };
}
__host__ __device__ inline float2 div(const float2& v, const float& f) {
	return { v.x / f, v.y / f };
}
__host__ __device__ inline float2 div(const float& f, const float2& v) {
	return { f / v.x, f / v.y };
}

__host__ __device__ inline float dot(const float2& v0, const float2& v1) {
	return v0.x * v1.x + v0.y * v1.y;
}
__host__ __device__ inline float dot(const float2& v) {
	return v.x * v.x + v.y * v.y;
}
__host__ __device__ inline float length(const float2& v) {
	return sqrtf(dot(v, v));
}
__device__ inline float invLength(const float2& v) {
	// return rsqrtf(dot(v, v));
	return 1.0f / sqrtf(dot(v, v));
}
__host__ inline float hinvLength(const float2& v) {
	return 1.0f / sqrtf(dot(v, v));
}
__host__ __device__ inline float sqrLength(const float2& v) {
	return dot(v, v);
}
__device__ inline float2 normalize(const float2& v) {
	float invl = invLength(v);
	return { v.x * invl, v.y * invl };
}
__host__ inline float2 hnormalize(const float2& v) {
	float invl = hinvLength(v);
	return { v.x * invl, v.y * invl };
}
__host__ __device__ inline float2 negate(const float2& v) {
	return { -v.x, -v.y };
}
__host__ __device__ inline float2 interpolate(const float2& v0, const float2& v1, const float& f) {
	return {
		v0.x * (1.0f - f) + v1.x * f,
		v0.y * (1.0f - f) + v1.y * f
	};
}

// float3

__host__ __device__ inline float3 add(const float3& v0, const float3& v1) {
	return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}
__host__ __device__ inline float3 add(const float3& v0, const float3& v1, const float3& v2) {
	return { v0.x + v1.x + v2.x, v0.y + v1.y + v2.y, v0.z + v1.z + v2.z };
}
__host__ __device__ inline float3 sub(const float3& v0, const float3& v1) {
	return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}
__host__ __device__ inline float3 mul(const float3& v0, const float3& v1) {
	return { v0.x * v1.x, v0.y * v1.y, v0.z * v1.z };
}
__host__ __device__ inline float3 div(const float3& v0, const float3& v1) {
	return { v0.x / v1.x, v0.y / v1.y, v0.z / v1.z };
}

__host__ __device__ inline float3 add(const float3& v, const float& f) {
	return { v.x + f, v.y + f, v.z + f };
}
__host__ __device__ inline float3 sub(const float3& v, const float& f) {
	return { v.x - f, v.y - f, v.z - f };
}
__host__ __device__ inline float3 mul(const float3& v, const float& f) {
	return { v.x * f, v.y * f, v.z * f };
}
__host__ __device__ inline float3 div(const float3& v, const float& f) {
	return { v.x / f, v.y / f, v.z / f };
}
__host__ __device__ inline float3 div(float f, const float3& v) {
	return { f / v.x, f / v.y, f / v.z };
}

__host__ __device__ inline float dot(const float3& v0, const float3& v1) {
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
__host__ __device__ inline float dot(const float3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}
__host__ __device__ inline float length(const float3& v) {
	return sqrtf(dot(v, v));
}
__device__ inline float invLength(const float3& v) {
	// return rsqrtf(dot(v, v));
	return 1.0f / sqrtf(dot(v, v));
}
__host__ inline float hinvLength(const float3& v) {
	return 1.0f / sqrtf(dot(v, v));
}
__host__ __device__ inline float sqrLength(const float3& v) {
	return dot(v, v);
}
__device__ inline float3 normalize(const float3& v) {
	float invl = invLength(v);
	return { v.x * invl, v.y * invl, v.z * invl };
}
__host__ inline float3 hnormalize(const float3& v) {
	float invl = hinvLength(v);
	return { v.x * invl, v.y * invl, v.z * invl };
}
__host__ __device__ inline float3 cross(const float3& v0, const float3& v1) {
	return {
		v0.y * v1.z - v0.z * v1.y,
		v0.z * v1.x - v0.x * v1.z,
		v0.x * v1.y - v0.y * v1.x
	};
}
__host__ __device__ inline float3 negate(const float3& v) {
	return { -v.x, -v.y, -v.z };
}
__host__ __device__ inline float3 interpolate(const float3& v0, const float3& v1, const float& f) {
	return {
		v0.x * (1.0f - f) + v1.x * f,
		v0.y * (1.0f - f) + v1.y * f,
		v0.z * (1.0f - f) + v1.z * f
	};
}

// float4

__host__ __device__ inline float4 add(const float4& v0, const float4& v1) {
	return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w };
}
__host__ __device__ inline float4 add(const float4& v0, const float4& v1, const float4& v2) {
	return { v0.x + v1.x + v2.x, v0.y + v1.y + v2.y, v0.z + v1.z + v2.z, v0.w + v1.w + v2.w };
}
__host__ __device__ inline float4 sub(const float4& v0, const float4& v1) {
	return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w };
}
__host__ __device__ inline float4 mul(const float4& v0, const float4& v1) {
	return { v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w };
}
__host__ __device__ inline float4 div(const float4& v0, const float4& v1) {
	return { v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w };
}

__host__ __device__ inline float4 add(const float4& v, const float& f) {
	return { v.x + f, v.y + f, v.z + f, v.w + f };
}
__host__ __device__ inline float4 sub(const float4& v, const float& f) {
	return { v.x - f, v.y - f, v.z - f, v.w - f };
}
__host__ __device__ inline float4 mul(const float4& v, const float& f) {
	return { v.x * f, v.y * f, v.z * f, v.w * f };
}
__host__ __device__ inline float4 div(const float4& v, const float& f) {
	return { v.x / f, v.y / f, v.z / f, v.w / f };
}
__host__ __device__ inline float4 div(const float& f, const float4& v) {
	return { f / v.x, f / v.y, f / v.z, f / v.w };
}

__host__ __device__ inline float dot(const float4& v0, const float4& v1) {
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}
__host__ __device__ inline float dot(const float4& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}
__host__ __device__ inline float length(const float4& v) {
	return sqrtf(dot(v, v));
}
__device__ inline float invLength(const float4& v) {
	// return rsqrtf(dot(v, v));
	return 1.0f / sqrtf(dot(v, v));
}
__host__ inline float hinvLength(const float4& v) {
	return 1.0f / sqrtf(dot(v, v));
}
__host__ __device__ inline float sqrLength(const float4& v) {
	return dot(v, v);
}
__device__ inline float4 normalize(const float4& v) {
	float invl = invLength(v);
	return { v.x * invl, v.y * invl, v.z * invl, v.w * invl };
}
__host__ inline float4 hnormalize(const float4& v) {
	float invl = hinvLength(v);
	return { v.x * invl, v.y * invl, v.z * invl, v.w * invl };
}
__host__ __device__ inline float4 negate(const float4& v) {
	return { -v.x, -v.y, -v.z, -v.w };
}
__host__ __device__ inline float4 interpolate(const float4& v0, const float4& v1, const float& f) {
	return {
		v0.x * (1.0f - f) + v1.x * f,
		v0.y * (1.0f - f) + v1.y * f,
		v0.z * (1.0f - f) + v1.z * f,
		v0.w * (1.0f - f) + v1.w * f
	};
}
