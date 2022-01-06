#include "application/application.h"

Application::Application(Window window)
    : window(window) {}

void Application::run() {

    initialize();
    
    while (!window.shouldClose()) {
        update();
        window.update();
    }

    terminate();
    window.close();

}