#include "graphics/camera.h"

#include <math.h>

#include <iostream>

#include "math/rt-math.cuh"

Camera::Camera(float3 position, float3 rotation)
    : position(position), rotation(rotation) { }
    
CameraController::CameraController(float vfov, Camera camera, float movementSpeed, float rotationSpeed, float3 up)
    : vfov(vfov), camera(camera), movementSpeed(movementSpeed), rotationSpeed(rotationSpeed), up(up) { }

void CameraController::updateCamera(Window* window, float dt) {

    float3 rot = mul(getRotation(window), dt);

    camera.position = add(camera.position, mul(getMovement(window), dt));
    camera.rotation = add(camera.rotation, rot);
    camera.rotation.x = fmaxf(-85, fminf(camera.rotation.x, 85));

    aspect = (float)window->getWidth() / (float)window->getHeight();
    float theta = vfov * (PI / 180.0f);
    float h = tanf(theta / 2.0f);
    float vpHeight = 2.0f * h;
    float vpWidth = aspect * vpHeight;

    direction = directionFromAngles(camera.rotation.x, camera.rotation.y);

    float3 w = hnormalize(negate(direction));
    float3 u = hnormalize(cross(up, w));
    float3 v = cross(w, u);

    camera.horizontal = mul(u, vpWidth);
    camera.vertical = mul(v, vpHeight);
    camera.lowerLeftCorner = sub(sub(sub(camera.position, div(camera.horizontal, 2.0f)), div(camera.vertical, 2.0f)), w);
}

float3 CameraController::getMovement(Window* window) {
    float2 inputXZ = make_float2(0, 0);
    float inputY = 0;

    if (window->input.getKey(GLFW_KEY_D))			inputXZ = add(inputXZ, make_float2( 1,  0));
    if (window->input.getKey(GLFW_KEY_A))			inputXZ = add(inputXZ, make_float2(-1,  0));
    if (window->input.getKey(GLFW_KEY_W))			inputXZ = add(inputXZ, make_float2( 0,  1));
    if (window->input.getKey(GLFW_KEY_S))			inputXZ = add(inputXZ, make_float2( 0, -1));
    if (window->input.getKey(GLFW_KEY_SPACE))		inputY += 1;
    if (window->input.getKey(GLFW_KEY_LEFT_SHIFT))  inputY -= 1;

    if (length(inputXZ) > 0.01f) {
        inputXZ = hnormalize(inputXZ);
    }

    float3 input = {
        inputXZ.x,
        inputY,
        inputXZ.y
    };
    
    input = mul(input, movementSpeed);

    float3 forward = direction;
    forward.y = 0;
    forward = hnormalize(forward);

    float3 toSide = cross(up, forward);
    toSide = hnormalize(toSide);

    float3 movement = add(
        mul(toSide, input.x),
        mul(up, input.y),
        mul(forward, input.z)
    );

    return movement;
}

float3 CameraController::getRotation(Window* window) {
    float3 rotation = make_float3(0, 0, 0);

    if (window->input.getKey(GLFW_KEY_UP))      rotation = add(rotation, make_float2( 1,  0));
    if (window->input.getKey(GLFW_KEY_DOWN))    rotation = add(rotation, make_float2(-1,  0));
    if (window->input.getKey(GLFW_KEY_RIGHT))   rotation = add(rotation, make_float2( 0,  1));
    if (window->input.getKey(GLFW_KEY_LEFT))    rotation = add(rotation, make_float2( 0, -1));

    return mul(rotation, rotationSpeed);
}