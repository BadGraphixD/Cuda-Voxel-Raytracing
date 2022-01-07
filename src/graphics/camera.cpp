#include "graphics/camera.h"

#include <math.h>

#include "math/rt_math.cuh"

void CameraController::updateCamera(const Window& window) {

    aspect = (float)window.getWidth() / (float)window.getHeight();
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