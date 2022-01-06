#pragma once

class Log {
public:
	static void Error(const char* msg);
	static void Warning(const char* msg);
	static void Message(const char* msg);
private:
	Log() {}
};