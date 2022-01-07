#pragma once

#include <cuda_runtime.h>

#include "graphics/window.h"

struct Camera {

    float3 position;
	float3 rotation;
    
	float3 lowerLeftCorner;
	float3 horizontal, vertical;

	Camera(float3 position = { 0, 0, 0 }, float3 rotation = { 0, 0, 0 })
	    : position(position), rotation(rotation) { }

};

class CameraController {
public:
	float vfov;
	float aspect;
	float3 direction;
	float3 up;

    Camera camera;
	
	CameraController() = default;
	CameraController(float vfov, Camera camera, float3 up = { 0, 1, 0 })
	    : vfov(vfov), up(up), camera(camera) { }

    void updateCamera(const Window& window);
};