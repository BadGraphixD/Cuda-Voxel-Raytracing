#pragma once

#include "application/application.h"
#include "util/render-data-displayer.h"
#include "util/grid-layout.h"
#include "util/buffer.cuh"

#include "graphics/raytracing/ray-generation.cuh"
#include "graphics/raytracing/ray-tracing.cuh"
#include "graphics/camera.h"

class CVRApp : public Application {
private:
    BufferManager bufferManager;
    RenderDataDisplayer displayer;

    GridLayout layout;

    CameraController cameraController;
    
    Buffer<Payload> d_payload;
    Buffer<Ray> d_rays;
    Buffer<uchar3> d_result;

protected:
    void initialize() override {
        bufferManager.addBuffer(&d_payload, BufferLocation::DEVICE);
        bufferManager.addBuffer(&d_rays, BufferLocation::DEVICE, window.getPixelAmount());
        bufferManager.addBuffer(&d_result, BufferLocation::DEVICE, window.getPixelAmount());

        layout = GridLayout({ 32, 32 }, window);
        cameraController = CameraController(90, Camera());
    }

    void update() override {

        cameraController.camera.rotation = add(cameraController.camera.rotation, { 0, .05f, 0 });
        cameraController.updateCamera(window);

        Payload payload = {
            window.getWidth(), window.getHeight(),
            cameraController.camera
        };
        
		d_payload.copyFrom(&payload, BufferLocation::HOST);

        generateRays<<<layout.blocks, layout.threads>>>(d_rays, d_payload);
        rayToUChar3<<<layout.blocks, layout.threads>>>(d_rays, d_result, d_payload);

        displayer.display(d_result.getPtr(), &window);
    }
    
    void terminate() override {
        bufferManager.destroyBuffers();
    }

public:
    CVRApp(Window window)
        : Application(window), displayer(&this->window) {}
};