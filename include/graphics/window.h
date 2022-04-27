#pragma once

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "util/input.h"

class Window {
private:
	GLFWwindow* window;
	GLuint width, height;

public:
	Input input;

	Window() = default;
	Window(GLuint width, GLuint height, const char* title);

	void update();
	bool shouldClose();
	void close();

	inline GLuint getWidth() const { return width; }
	inline GLuint getHeight() const { return height; }
	inline GLuint getPixelAmount() const { return width * height; }

	void setWindowUserPoiner();

private:
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void errorCallback(int error, const char* msg);
};
