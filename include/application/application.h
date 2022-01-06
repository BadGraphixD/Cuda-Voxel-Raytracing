#pragma once

#include "graphics/window.h"

class Application {
protected:
    Window window;

    virtual void initialize() = 0;
    virtual void update() = 0;
    virtual void terminate() = 0;

public:
    Application(Window window);
    void run();

};