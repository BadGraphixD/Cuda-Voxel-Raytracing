#include "util/input.h"

bool Input::getKey(GLuint key) const {
    return keys[key];
}

bool Input::getButton(GLuint button) const {
    return buttons[button];
}