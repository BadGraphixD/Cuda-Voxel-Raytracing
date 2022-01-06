#include "util/log.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

static std::string date() {

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    return oss.str();
}

void Log::Error(const char* msg) {
    printf("[%s] Error: %s\n", date().c_str(), msg);
}

void Log::Warning(const char* msg) {
    printf("[%s] Warning: %s\n", date().c_str(), msg);
}

void Log::Message(const char* msg) {
    printf("[%s] %s\n", date().c_str(), msg);
}