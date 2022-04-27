#include "application/application.h"

Application::Application(Window window)
    : window(window) { }

void Application::run() {
    
    window.setWindowUserPoiner();
    initialize();
    
    lastFrameTime = glfwGetTime();
    
    while (!window.shouldClose()) {

        currentFrameTime = glfwGetTime();
        deltaTime = (float)(currentFrameTime - lastFrameTime);
        lastFrameTime = currentFrameTime;

        update();
        window.update();
        
        frame++;
    }

    terminate();
    window.close();

}