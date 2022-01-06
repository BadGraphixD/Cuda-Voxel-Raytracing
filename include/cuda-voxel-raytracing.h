#pragma once

#include "application/application.h"
#include "util/buffer.cuh"
#include "util/render-data-displayer.h"

class CVRApp : public Application {
private:
    BufferManager bufferManager;
    RenderDataDisplayer displayer;

    Buffer<uchar3> buffer;

protected:
    void initialize() override {
        bufferManager.addBuffer(&buffer, BufferLocation::DEVICE, window.getPixelAmount());
        buffer.fill(128);
    }

    void update() override {
        displayer.display(buffer.getPtr(), &window);
    }
    
    void terminate() override {
        bufferManager.destroyBuffers();
    }

public:
    CVRApp(Window window)
        : Application(window), displayer(&this->window) {}
};