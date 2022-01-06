#include "graphics/window.h"

#include "util/log.h"

Window::Window(GLuint width, GLuint height, const char* title)
    : width(width), height(height) {

    if (!glfwInit()) {
        Log::Error("Failed to initialize GLFW!");
        return;
    }

    window = glfwCreateWindow(width, height, title, 0, 0);
    if (!window) {
        Log::Error("Failed to create GLFW window!");
        return;
    }

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetErrorCallback(errorCallback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
}

void Window::update() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Window::shouldClose() {
    return glfwWindowShouldClose(window);
}

void Window::close() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // ((Window*)glfwGetWindowUserPointer(window))->input->keys[key] = action != GLFW_RELEASE; // TODO: fix invalid ptr bug
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    // ((Window*)glfwGetWindowUserPointer(window))->input->buttons[button] = action != GLFW_RELEASE; // TODO: fix invalid ptr bug
}

void Window::errorCallback(int error, const char* msg) {
    Log::Error(msg);
}

void Window::setWindowUserPoiner() {
    glfwSetWindowUserPointer(window, this);
}