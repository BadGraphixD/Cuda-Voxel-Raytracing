#pragma once

#include "graphics/window.h"

class Application {
protected:
    Window window;

    virtual void initialize() = 0;
    virtual void update() = 0;
    virtual void terminate() = 0;
    
	float deltaTime;
	int frame = 0;

private:
	double currentFrameTime;
	double lastFrameTime;

public:
    Application(Window window);
    void run();

};