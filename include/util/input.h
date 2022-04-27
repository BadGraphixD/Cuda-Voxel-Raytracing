#pragma once

#include "GL/glew.h"
#include "GLFW/glfw3.h"

struct Input {
	bool keys[GLFW_KEY_LAST] = { false };
	bool buttons[GLFW_MOUSE_BUTTON_LAST] = { false };

	Input() = default;

	bool getKey(GLuint key) const;
	bool getButton(GLuint button) const;
};