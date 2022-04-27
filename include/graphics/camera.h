#pragma once

#include <cuda_runtime.h>

#include "graphics/window.h"

struct Camera {
    float3 position;
	float3 rotation;
    
	float3 lowerLeftCorner;
	float3 horizontal, vertical;

	Camera(float3 position = make_float3(0, 0, 0), float3 rotation = make_float3(0, 0, 0));
};

class CameraController {
public:
	CameraController() = default;
	CameraController(float vfov, Camera camera, float movementSpeed = 1.0f, float rotationSpeed = 1.0f, float3 up = make_float3(0, 1, 0));
    void updateCamera(Window* window, float dt);

	// todo: remove bug and this method
	void tempInitializeBecauseOfBug(Window* window, float dt) {
        updateCamera(window, dt);
        camera.position = make_float3(.5, 2, .5);
        camera.rotation = make_float3(-80, 0, 0);
	}

	inline Camera getCamera() { return camera; }

private:
	float vfov;
	float aspect;
	float3 direction;
	float3 up;

	float movementSpeed;
	float rotationSpeed;
	
    Camera camera;

	float3 getMovement(Window* window);
	float3 getRotation(Window* window);
};